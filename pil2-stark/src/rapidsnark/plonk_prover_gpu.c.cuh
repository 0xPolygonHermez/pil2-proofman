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
typedef void (*FileReadFn)(void* dest, uint32_t sectionId, uint64_t offset, uint64_t len, void* ctx);
extern "C" void msm_bn128_gpu_dev_ptr(void* out, const void* d_points, const void* d_scalars, size_t npoints, bool montgomery);
extern "C" void ntt_bn128_gpu_dev_ptr(void* d_data, uint32_t lg_n);
extern "C" void intt_bn128_gpu_dev_ptr(void* d_data, uint32_t lg_n);
extern "C" void gpu_plonk_memcpy_h2d(void* dst, const void* src, size_t bytes);
extern "C" void gpu_plonk_memcpy_d2h(void* dst, const void* src, size_t bytes);
extern "C" void gpu_plonk_memcpy_d2d(void* dst, const void* src, size_t bytes);
extern "C" void gpu_plonk_compute_pi(
    void* piOut,
    FileReadFn readFn, void* readCtx,
    uint32_t lagrangeSectionId,
    uint64_t lagrangeBaseOffset,
    uint64_t lagrangeStride,
    const void* publicA,
    uint64_t NExt, uint32_t nPublic,
    void* dPI, void* dLag,
    void* pinnedBuf, size_t pinnedSize);
extern "C" void gpu_plonk_compute_pi_single(
    void* dPI, const void* dLag,
    const void* publicVal, uint64_t NExt);
extern "C" void gpu_plonk_cuda_malloc(void** dBuffer, uint64_t buffeSize);
extern "C" void gpu_plonk_cuda_free(void* dBuffer);
extern "C" void gpu_plonk_cuda_malloc_pinned_buffer(void** pinnedBuffer, size_t pinnedSize);
extern "C" void gpu_plonk_free_pinned_buffer(void* pinnedBuffer);
extern "C" void gpu_plonk_start_static_eval_transfer(
    FileReadFn readFn, void* readCtx,
    void* dBuffer, void* pinnedBuffer, size_t pinnedSize,
    const uint32_t* sectionIds, const uint64_t* byteOffsets, const uint64_t* byteSizes, int numArrays);
extern "C" void gpu_plonk_start_cpu_to_gpu_transfer(
    void** dDsts, const void** hostSrcs, const size_t* sizes, int numArrays,
    void* pinnedBuffer, size_t pinnedSize);
extern "C" void gpu_plonk_zero_pad(void* buf, uint64_t startElem, uint64_t endElem);
extern "C" void gpu_plonk_zero_pad_async(void* buf, uint64_t startElem, uint64_t endElem, void* stream);
extern "C" void gpu_plonk_compute_gate_a(
    void* tOut, void* tzOut,
    const void* evalA, const void* evalQL,
    const void* d_blindings,
    uint64_t N,
    const void* omegaBases, const void* omegaTid);
extern "C" void gpu_plonk_compute_gate_b(
    void* tOut, void* tzOut,
    const void* evalB, const void* evalQR,
    const void* d_blindings,
    uint64_t N,
    const void* omegaBases, const void* omegaTid);
extern "C" void gpu_plonk_compute_gate_c(
    void* tOut, void* tzOut,
    const void* evalC, const void* evalQO, const void* evalQC,
    const void* d_blindings,
    uint64_t N,
    const void* omegaBases, const void* omegaTid);

extern "C" void gpu_plonk_precompute_omega_tables_async(
    void* dBases, void* dTid, const void* omega4xPtr,
    uint32_t blockSize, uint32_t numBlocks, void* stream);
extern "C" void gpu_plonk_cuda_device_sync();
extern "C" void* gpu_plonk_create_cuda_stream_nonblocking();
extern "C" void gpu_plonk_destroy_cuda_stream(void* stream);
extern "C" void gpu_plonk_sync_cuda_stream(void* stream);
extern "C" void gpu_plonk_memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream);
extern "C" void gpu_plonk_pin_host_memory(void* ptr, size_t bytes);
extern "C" void gpu_plonk_unpin_host_memory(void* ptr);
extern "C" void gpu_plonk_compute_r_wxi(
    void* wxi,
    const void* polA, const void* polB, const void* polC, const void* polZ,
    const void* polQM, const void* polQL, const void* polQR, const void* polQO,
    const void* polQC,
    const void* polS1, const void* polS2, const void* polS3,
    const void* polT1, const void* polT2, const void* polT3,
    const void* constants, uint64_t N);
extern "C" void gpu_plonk_gather_witness(
    void* evalOut, const void* mapBuffer,
    const void* witness, const void* intWitness,
    uint32_t nDirect, uint64_t nConstraints, uint64_t N);
extern "C" void gpu_plonk_compute_z_ratios_gather(
    void* ratioOut,
    const void* mapA, const void* mapB, const void* mapC,
    const void* witness, const void* intWitness,
    uint32_t nDirect, uint64_t nConstraints,
    const void* dStaticEvals,
    const void* betaPtr, const void* gammaPtr,
    const void* k1Ptr, const void* k2Ptr,
    uint64_t N,
    const void* omegaBases, const void* omegaTid);
extern "C" void gpu_plonk_prefix_scan_multiply(void* dData, uint64_t N, void* dWork);
extern "C" void gpu_plonk_rotate_left(void* dst, const void* src, uint64_t N);
extern "C" void gpu_plonk_compute_div_zerofier(
    void* dCoefs, uint64_t length,
    const void* invBetaPtr, const void* y0Ptr, void* dPairWork);
extern "C" void gpu_plonk_divzh_add(void* dT, const void* dTz, uint64_t N);
extern "C" void gpu_plonk_poly_eval_to_host(
    void* hostResult, const void* coefs, const void* pointPtr,
    uint64_t N, void* dWork);
extern "C" void gpu_plonk_split_t_blinding(
    void* d_t1, void* d_t2, void* d_t3,
    const void* dTcombined, const void* d_blindings, uint64_t N);
extern "C" void gpu_plonk_compute_qm_perm_l1(
    void* tOut, void* tzOut,
    const void* evalA, const void* evalB, const void* evalC,
    const void* evalZ,
    const void* evalQM, const void* evalS1, const void* evalS2, const void* evalS3, const void* evalL1,
    const void* d_blindings, const void* d_zvals,
    const void* beta, const void* gamma,
    const void* alpha, const void* alpha2,
    const void* k1, const void* k2,
    const void* omega1,
    uint64_t N,
    const void* omegaBases, const void* omegaTid);

extern "C" void gpu_plonk_calculate_additions(
    void* d_buffInternalWitness,
    const void* d_buffWitness,
    const void* d_addSignalId1,
    const void* d_addSignalId2,
    const void* d_addFactor1,
    const void* d_addFactor2,
    const void* d_additionLevels,
    uint8_t maxLevel,
    uint32_t nAdditions,
    uint32_t nDirect);

namespace PlonkGPU
{
    template <typename Engine>
    void PlonkProverGPU<Engine>::initialize(void *reservedMemoryPtr, uint64_t reservedMemorySize)
    {
        zkey = NULL;
        this->reservedMemoryPtr = (FrElement *)reservedMemoryPtr;
        this->reservedMemorySize = reservedMemorySize;

        // Initialize all dynamic pointers to nullptr to ensure safe deletion
        precomputedBigBuffer = nullptr;
        nonPrecomputedBigBuffer = nullptr;
        mapBuffersBigBuffer = nullptr;
        buffInternalWitness = nullptr;
        additionsBuff = nullptr;
        additionLevels = nullptr;
        maxAdditionLevel = 0;
        fft = nullptr;
        transcript = nullptr;
        proof = nullptr;
        isMyDeviceBuffer = false;
        evalConstPols = true; //this will be the default harcoded value

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
        // Unpin precomputedBigBuffer before deleting
        if (precomputedPinned && precomputedBigBuffer) {
            gpu_plonk_unpin_host_memory(precomputedBigBuffer);
            precomputedPinned = false;
        }

        // DELETE RESERVED MEMORY (if necessary)
        delete[] precomputedBigBuffer;
        precomputedBigBuffer = nullptr;
        delete[] mapBuffersBigBuffer;
        mapBuffersBigBuffer = nullptr;
        delete[] buffInternalWitness;
        buffInternalWitness = nullptr;
        delete[] additionsBuff;
        additionsBuff = nullptr;
        delete[] additionLevels;
        additionLevels = nullptr;

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
            delete zkey;
            zkey = nullptr;
        }

        if (NULL == reservedMemoryPtr)
        {
            delete[] nonPrecomputedBigBuffer;
            nonPrecomputedBigBuffer = nullptr;
        }

        if (pTauStream) {
            gpu_plonk_destroy_cuda_stream(pTauStream);
            pTauStream = nullptr;
        }

        if(omegasStream) {
            gpu_plonk_destroy_cuda_stream(omegasStream);
            omegasStream = nullptr;
        }

        if (evalNTTStream) {
            gpu_plonk_destroy_cuda_stream(evalNTTStream);
            evalNTTStream = nullptr;
        }

        if (pinnedD2HStaging) {
            gpu_plonk_free_pinned_buffer(pinnedD2HStaging);
            pinnedD2HStaging = nullptr;
        }

        if (d_unifiedBuffer) {
            if(isMyDeviceBuffer) {
                gpu_plonk_cuda_free(d_unifiedBuffer);
            }
            d_unifiedBuffer = nullptr;
            d_staticEvalsBuffer = nullptr;
            d_piBuffer = nullptr;
            d_lagBuffer = nullptr;
            d_evalsT = nullptr;
            d_evalsTz = nullptr;
            d_ptau = nullptr;
            d_polCoefA = nullptr;
            d_polCoefB = nullptr;
            d_polCoefC = nullptr;
            d_polCoefZ = nullptr;
            d_aux = nullptr;
            d_gathered = nullptr;
            d_ratios = nullptr;
            d_zEvals = nullptr;
            d_scanWork = nullptr;            
            d_omegaBasesN = nullptr;
            d_omegaTidN = nullptr;
            d_blindings = nullptr;
            d_zvals = nullptr;
            d_omegaBasesNExt = nullptr;
            d_omegaTidNExt = nullptr;
            d_witness = nullptr;
            d_intWitness = nullptr;
            d_mapBuffers = nullptr;
            d_coefQM = nullptr;
            d_coefQL = nullptr;
            d_coefQR = nullptr;
            d_coefQO = nullptr;
            d_coefQC = nullptr;
            d_coefS1 = nullptr;
            d_coefS2 = nullptr;
            d_coefS3 = nullptr;
            d_t1 = nullptr;
            d_t2 = nullptr;
            d_t3 = nullptr;
            d_evalsS1 = nullptr;
            d_evalsS2 = nullptr;
            d_evalsS3 = nullptr;
            d_evalsL1 = nullptr;
            d_evalsQL = nullptr;
            d_evalsQR = nullptr;
            d_evalsQM = nullptr;
            d_evalsQO = nullptr;
            d_evalsQC = nullptr;
            d_evalsA = nullptr;
            d_evalsB = nullptr;
            d_evalsC = nullptr;
        } else {
            gpu_plonk_cuda_free(d_staticEvalsBuffer);
            gpu_plonk_cuda_free(d_piBuffer);
            gpu_plonk_cuda_free(d_lagBuffer);
            gpu_plonk_cuda_free(d_ptau);
            d_staticEvalsBuffer = nullptr;
            d_piBuffer = nullptr;
            d_lagBuffer = nullptr;
            d_ptau = nullptr;
        }
        gpu_plonk_free_pinned_buffer(pinnedQ);
        gpu_plonk_free_pinned_buffer(pinnedS);
        gpu_plonk_free_pinned_buffer(pinnedPI);
        pinnedQ = nullptr;
        pinnedS = nullptr;
        pinnedPI = nullptr;

        delete fft;
        fft = nullptr;

        mapBuffers.clear();
        
        polPtr.clear();
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

        N = zkey->domainSize;
        NBytes = N * sizeof(FrElement);
        NExt = 4 * N;
        NExtBytes = NExt * sizeof(FrElement);

        LOG_TRACE("> Starting fft");

        fft = new FFT<typename Engine::Fr>(NExt);
        zkeyPower = fft->log2(N);

        mpz_t altBbn128r;
        mpz_init(altBbn128r);
        mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);

        if (mpz_cmp(zkey->rPrime, altBbn128r) != 0)
        {
            throw std::invalid_argument("zkey curve not supported");
        }

        mpz_clear(altBbn128r);

        

        ////////////////////////////////////////////////////
        // PRECOMPUTED BIG BUFFER
        ////////////////////////////////////////////////////
        uint64_t lengthPrecomputedBigBuffer = 0;
        lengthPrecomputedBigBuffer += N * 1 * 9; // S1-S3, QL-QC, L1
        lengthPrecomputedBigBuffer += (N + 6) * sizeof(G1PointAffine) / sizeof(FrElement); // PTau buffer

        precomputedBigBuffer = new FrElement[lengthPrecomputedBigBuffer];

        polPtr["Sigma1"] = &precomputedBigBuffer[0];
        polPtr["Sigma2"] = polPtr["Sigma1"] + N;
        polPtr["Sigma3"] = polPtr["Sigma2"] + N;
        polPtr["QL"] = polPtr["Sigma3"] + N;
        polPtr["QR"] = polPtr["QL"] + N;
        polPtr["QM"] = polPtr["QR"] + N;
        polPtr["QO"] = polPtr["QM"] + N;
        polPtr["QC"] = polPtr["QO"] + N;
        polPtr["L1"] = polPtr["QC"] + N;

        PTau = (G1PointAffine *)(polPtr["L1"] + N);

        LOG_TRACE("... Loading QL, QR, QM, QO, & QC polynomial coefficients");

        int nThreads = omp_get_max_threads() / 2;

        if (dr) {
            fdZkey->readSectionToParallel(polPtr["QL"], Zkey::ZKEY_PL_QL_SECTION, 0, NBytes);
            fdZkey->readSectionToParallel(polPtr["QR"], Zkey::ZKEY_PL_QR_SECTION, 0, NBytes);
            fdZkey->readSectionToParallel(polPtr["QM"], Zkey::ZKEY_PL_QM_SECTION, 0, NBytes);
            fdZkey->readSectionToParallel(polPtr["QO"], Zkey::ZKEY_PL_QO_SECTION, 0, NBytes);
            fdZkey->readSectionToParallel(polPtr["QC"], Zkey::ZKEY_PL_QC_SECTION, 0, NBytes);
        } else {
            ThreadUtils::parcpy(polPtr["QL"],
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QL_SECTION),
                                NBytes, nThreads);
            ThreadUtils::parcpy(polPtr["QR"],
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QR_SECTION),
                                NBytes, nThreads);
            ThreadUtils::parcpy(polPtr["QM"],
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QM_SECTION),
                                NBytes, nThreads);
            ThreadUtils::parcpy(polPtr["QO"],
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QO_SECTION),
                                NBytes, nThreads);
            ThreadUtils::parcpy(polPtr["QC"],
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QC_SECTION),
                                NBytes, nThreads);
        }

        LOG_TRACE("... Loading Sigma1, Sigma2 & Sigma3 polynomial coefficients");

        if (dr) {
            fdZkey->readSectionToParallel(polPtr["Sigma1"], Zkey::ZKEY_PL_SIGMA_SECTION, 0, NBytes);
            fdZkey->readSectionToParallel(polPtr["Sigma2"], Zkey::ZKEY_PL_SIGMA_SECTION, NBytes * 5, NBytes);
            fdZkey->readSectionToParallel(polPtr["Sigma3"], Zkey::ZKEY_PL_SIGMA_SECTION, NBytes * 10, NBytes);
        } else {
            ThreadUtils::parcpy(polPtr["Sigma1"],
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_SIGMA_SECTION),
                                NBytes, nThreads);
            ThreadUtils::parcpy(polPtr["Sigma2"],
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_SIGMA_SECTION) + NBytes * 5,
                                NBytes, nThreads);
            ThreadUtils::parcpy(polPtr["Sigma3"],
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_SIGMA_SECTION) + NBytes * 10,
                                NBytes, nThreads);
        }

        LOG_TRACE("... Loading L1 polynomial coefficients");
        if (zkey->nPublic != 1) {
            throw std::runtime_error("GPU PLONK prover currently requires exactly 1 public input (nPublic=" + std::to_string(zkey->nPublic) + ")");
        }
        if (dr) {
            fdZkey->readSectionToParallel(polPtr["L1"], Zkey::ZKEY_PL_LAGRANGE_SECTION, 0, NBytes);
        } else {
            ThreadUtils::parcpy(polPtr["L1"],
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_LAGRANGE_SECTION),
                                NBytes, nThreads);
        }

        LOG_TRACE("... Loading Powers of Tau evaluations");
        if (dr) {
            fdZkey->readSectionToParallel(this->PTau, Zkey::ZKEY_PL_PTAU_SECTION, 0,
                                  (N + 6) * sizeof(G1PointAffine));
        } else {
            ThreadUtils::parcpy(this->PTau,
                                (G1PointAffine *)fdZkey->getSectionData(Zkey::ZKEY_PL_PTAU_SECTION),
                                (N + 6) * sizeof(G1PointAffine), nThreads);
        }

        // Load A, B & C map buffers
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

        LOG_TRACE("··· Precomputing addition dependency levels");
        additionLevels = new uint8_t[zkey->nAdditions];
        uint32_t nDirect = zkey->nVars - zkey->nAdditions;
        maxAdditionLevel = 0;
        for (uint32_t i = 0; i < zkey->nAdditions; i++) {
            uint8_t l = 0;
            if (additionsBuff[i].signalId1 >= nDirect)
                l = std::max(l, (uint8_t)(additionLevels[additionsBuff[i].signalId1 - nDirect] + 1));
            if (additionsBuff[i].signalId2 >= nDirect)
                l = std::max(l, (uint8_t)(additionLevels[additionsBuff[i].signalId2 - nDirect] + 1));
            additionLevels[i] = l;
            maxAdditionLevel = std::max(maxAdditionLevel, l);
        }
        assert(maxAdditionLevel < 255);

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

        ////////////////////////////////////////////////////
        // NON-PRECOMPUTED BIG BUFFER
        ////////////////////////////////////////////////////
        // GPU path: all polynomial/eval work (T, Tz, A, B, C, Z, T1, T2, T3, R, Wxi, PI)
        // happens on GPU. Only buffers["publics"][0..nPublic) survives for transcript + async PI.
        lengthNonPrecomputedBigBuffer = zkey->nPublic;

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

        buffers["publics"] = &nonPrecomputedBigBuffer[0];

        // Compute static eval buffer size (GPU alloc deferred to round0 as part of unified buffer)
        pinnedSize = 512ULL * 1024 * 1024;  // 512 MB per thread
        gpu_plonk_cuda_malloc_pinned_buffer(&pinnedQ, pinnedSize);
        gpu_plonk_cuda_malloc_pinned_buffer(&pinnedS, pinnedSize);
        gpu_plonk_cuda_malloc_pinned_buffer(&pinnedPI, pinnedSize);

        // Pin precomputedBigBuffer for async H2D of zkey coef polys (round3) and PTau (round0)
        if (precomputedBigBuffer && !precomputedPinned) {
            size_t pinBytes = 9 * N * sizeof(FrElement)  // 9 poly coef arrays (S1-S3, QL-QC, L1)
                            + (N + 6) * sizeof(G1PointAffine); // PTau
            gpu_plonk_pin_host_memory(precomputedBigBuffer, pinBytes);
            precomputedPinned = true;
        }

        // Store file pointer for file-based GPU transfer in round0
        fdZkeyPtr = fdZkey;
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::preAllocate(void* unified_buffer_gpu)
    {
        LOG_TRACE("> Pre-allocating GPU static eval buffers (round0)");
        d_unifiedBuffer = unified_buffer_gpu;
        round0();
        preAllocated = true;
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
        ss << "  Domain size:   " << N;
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

        uint32_t nDirect = zkey->nVars - zkey->nAdditions;
        LOG_TRACE("> Uploading witness to GPU");
        gpu_plonk_memcpy_h2d(d_witness, buffWitness, nDirect * sizeof(FrElement));

        // Until this point all calculations made are circuit depending and independent from the data,
        // from here is the proof calculation

        double startTime = omp_get_wtime();

        // ROUND 0
        if (!preAllocated) {
            LOG_TRACE("> ROUND 0 (load data to GPU)");
#ifdef PLONK_GPU_TIMING
            double tSetup = omp_get_wtime();
#endif
                round0();
#ifdef PLONK_GPU_TIMING
            std::cout << "[SETUP] round0: " << omp_get_wtime() - tSetup << "s" << std::endl;
#endif
        } else {
            LOG_TRACE("> ROUND 0 skipped (pre-allocated)");
        }
        preAllocated = false;

        LOG_TRACE("> Computing Additions");
#ifdef PLONK_GPU_TIMING
        double tSetup = omp_get_wtime();
#endif
        calculateAdditions();
#ifdef PLONK_GPU_TIMING
        std::cout << "[SETUP] calculateAdditions (" << zkey->nAdditions << " additions): " << omp_get_wtime() - tSetup << "s" << std::endl;
#endif

        // Fill publics buffer on CPU for transcript (needed by round1)
        if (evalConstPols) {
            for (uint32_t i = 0; i < zkey->nPublic; i++) {
                FrElement w = getWitness(mapBuffers["A"][i]);
                E.fr.toMontgomery(buffers["publics"][i], w);
            }
        }

        // Wait for async eval NTT (launched in round0/preAllocate) to complete.
        // Must join before round1: to avoid concurrent calls from asyncEvalNTT and computeWirePolynomials
        if (evalConstPols && asyncEvalNTT.joinable()) {
#ifdef PLONK_GPU_TIMING
            double t0 = omp_get_wtime();
#endif
            asyncEvalNTT.join();
            gpu_plonk_cuda_device_sync();
#ifdef PLONK_GPU_TIMING
            std::cout << "[SETUP] eval NTT join waited: " << omp_get_wtime() - t0 << "s" << std::endl;
#endif
        }

        // Compute PI(X) — L1 evaluations already on GPU in d_evalsL1 from asyncEvalNTT
        if (evalConstPols) {
            LOG_TRACE("> Computing PI (simplified, nPublic=1)");
#ifdef PLONK_GPU_TIMING
            double tPI = omp_get_wtime();
#endif
            gpu_plonk_compute_pi_single(d_piBuffer, d_evalsL1,
                                         &buffers["publics"][0], NExt);
            gpu_plonk_cuda_device_sync();
#ifdef PLONK_GPU_TIMING
            std::cout << "[SETUP] compute PI (single): " << omp_get_wtime() - tPI << "s" << std::endl;
#endif
        }

        // START PLONK PROVER PROTOCOL

        // ROUND 1. Compute C1(X) polynomial
        LOG_TRACE("> ROUND 1");
#ifdef PLONK_GPU_TIMING
        double tRound = omp_get_wtime();
#endif
        round1();
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING] Round 1 (wire polys + commits): " << omp_get_wtime() - tRound << "s" << std::endl;
#endif

        // ROUND 2. Compute C2(X) polynomial
        LOG_TRACE("> ROUND 2");
#ifdef PLONK_GPU_TIMING
        tRound = omp_get_wtime();
#endif
        round2();
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING] Round 2 (Z poly + commit): " << omp_get_wtime() - tRound << "s" << std::endl;
#endif

        // ROUND 3. Compute opening evaluations
        LOG_TRACE("> ROUND 3");
#ifdef PLONK_GPU_TIMING
        tRound = omp_get_wtime();
#endif
        round3();
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING] Round 3 (T poly + commits): " << omp_get_wtime() - tRound << "s" << std::endl;
#endif

        // ROUND 4. Compute W(X) polynomial
        LOG_TRACE("> ROUND 4");
#ifdef PLONK_GPU_TIMING
        tRound = omp_get_wtime();
#endif
        round4();
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING] Round 4 (evaluations): " << omp_get_wtime() - tRound << "s" << std::endl;
#endif

        // ROUND 5. Compute W'(X) polynomial
        LOG_TRACE("> ROUND 5");
#ifdef PLONK_GPU_TIMING
        tRound = omp_get_wtime();
#endif
        round5();
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING] Round 5 (linearization + commits): " << omp_get_wtime() - tRound << "s" << std::endl;
#endif

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
        uint32_t nDirect = zkey->nVars - zkey->nAdditions;

        gpu_plonk_calculate_additions(
            d_intWitness,              // output: internal witness results
            d_witness,
            d_addSignalId1,
            d_addSignalId2,
            d_addFactor1,
            d_addFactor2,
            d_additionLevels,
            maxAdditionLevel,
            zkey->nAdditions,
            nDirect
        );
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

    // ROUND 0 — Populate static eval arrays (file I/O or NTT from coefficients)
    template <typename Engine>
    void PlonkProverGPU<Engine>::round0()
    {
        uint64_t ptauBytes = (N + 6) * sizeof(G1PointAffine);
        uint32_t numBlocks4N = (uint32_t)(NExt / 256);
        uint32_t numBlocksN = (uint32_t)(N / 256);
        uint64_t scanWorkElems = N / 1024 + N / (1024 * 1024) + 1; // mulScan workspace (round2): recursive blocks of 1024

        // d_aux peak usage per round:
        //   Round 1-3 (NTT): NExt + 4 FrElements (4N evals + Z wrap-around)
        //   Round 2 (prefix scan): 4N + scanWorkElems
        //   Round 5 (divByZerofier): (N+6) + 2*(N+5) + affineScanWork FrElements (~3N)
        uint64_t auxElemsR3 = NExt + 4 + scanWorkElems;  // Round 1-3: NTT + Z wrap-around + scan workspace
        uint64_t affineScanPairs = N + 5;  // numPairs for Wxi (largest case)
        uint64_t affineScanWork = 0;
        for (uint64_t n = affineScanPairs; n > 1; n = (n + 1023) / 1024)
            affineScanWork += n;
        uint64_t auxElemsR5 = (N + 6) + affineScanWork * 2;
        uint64_t auxElems = std::max(auxElemsR3, auxElemsR5);
        uint32_t nDirect = zkey->nVars - zkey->nAdditions;

        if (!d_unifiedBuffer) {

            uint64_t unifiedBufferSize = 0;
            unifiedBufferSize += 9 * NExtBytes;                       // 9 static eval arrays (S1,S2,S3,L1,QL,QR,QM,QO,QC)
            unifiedBufferSize += NExtBytes;                           // PI / T buffer
            unifiedBufferSize += NExtBytes;                           // Lag / Tz buffer
            unifiedBufferSize += 4 * NBytes;                          // PolCoefA/B/C/Z
            unifiedBufferSize += ptauBytes;                           // PTau
            unifiedBufferSize = (unifiedBufferSize + 255) & ~255ULL;  // align Aux to 256 bytes
            unifiedBufferSize += auxElems * sizeof(FrElement);        // Aux scratch: max(round3 NTT, round5 affineScan)
            unifiedBufferSize += numBlocks4N * sizeof(FrElement);     // omega_4x bases for 4N gate kernels
            unifiedBufferSize += 256 * sizeof(FrElement);             // omega_4x tid table (0..255)
            unifiedBufferSize += numBlocksN * sizeof(FrElement);      // omega bases for N z_ratios kernel
            unifiedBufferSize += 256 * sizeof(FrElement);             // omega tid table (0..255)
            unifiedBufferSize += (BLINDINGFACTORSLENGTH_PLONK_GPU + 1) * sizeof(FrElement); // blinding factors b_0..b_11
            unifiedBufferSize += 12 * sizeof(FrElement);                                    // Z1[4], Z2[4], Z3[4] for QM+perm+L1

            // GPU additions data - grouped by type for proper alignment (uint32_t at the end)
            unifiedBufferSize += nDirect * sizeof(FrElement);          // primary witnesses
            unifiedBufferSize += zkey->nAdditions * sizeof(FrElement); // internal witnesses
            unifiedBufferSize += zkey->nAdditions * sizeof(FrElement); // addFactor1
            unifiedBufferSize += zkey->nAdditions * sizeof(FrElement); // addFactor2
            unifiedBufferSize += zkey->nAdditions * sizeof(uint32_t);  // addSignalId1
            unifiedBufferSize += zkey->nAdditions * sizeof(uint32_t);  // addSignalId2
            unifiedBufferSize += zkey->nAdditions;                     // additionLevels (uint8_t per addition)

            isMyDeviceBuffer = true;
            gpu_plonk_cuda_malloc(&d_unifiedBuffer, unifiedBufferSize);
            //std::cout << "[round0] Unified GPU buffer: " << unifiedBufferSize / (1024.0*1024*1024) << " GiB" << std::endl;
        }

        // Set all pointers as offsets into unified buffer
        uint8_t* base = (uint8_t*)d_unifiedBuffer;
        uint64_t off = 0;
        d_staticEvalsBuffer = base + off;    off += 9 * NExtBytes;
        d_piBuffer = base + off;             off += NExtBytes;
        d_lagBuffer = base + off;            off += NExtBytes;
        d_polCoefA = base + off;             off += NBytes;
        d_polCoefB = base + off;             off += NBytes;
        d_polCoefC = base + off;             off += NBytes;
        d_polCoefZ = base + off;             off += NBytes;
        d_ptau = base + off;                 off += ptauBytes;
        off = (off + 255) & ~255ULL;
        d_aux = base + off;                  off += auxElems * sizeof(FrElement);
        d_omegaBasesNExt = base + off;       off += numBlocks4N * sizeof(FrElement);
        d_omegaTidNExt = base + off;         off += 256 * sizeof(FrElement);
        d_omegaBasesN = base + off;          off += numBlocksN * sizeof(FrElement);
        d_omegaTidN = base + off;            off += 256 * sizeof(FrElement);
        d_blindings = base + off;            off += (BLINDINGFACTORSLENGTH_PLONK_GPU + 1) * sizeof(FrElement);
        d_zvals = base + off;                off += 12 * sizeof(FrElement);
        d_witness = base + off;              off += nDirect * sizeof(FrElement);
        d_intWitness = base + off;           off += zkey->nAdditions * sizeof(FrElement);
        d_addFactor1 = base + off;           off += zkey->nAdditions * sizeof(FrElement);
        d_addFactor2 = base + off;           off += zkey->nAdditions * sizeof(FrElement);
        d_addSignalId1 = base + off;         off += zkey->nAdditions * sizeof(uint32_t);
        d_addSignalId2 = base + off;         off += zkey->nAdditions * sizeof(uint32_t);
        d_additionLevels = base + off;       off += zkey->nAdditions;

        // Derived aliases within d_aux
        d_gathered   = d_aux;                                                         // [0..N) gather output / IFFT (round1)
        d_ratios     = d_aux;                                                         // [0..N) z_ratios / prefix scan (round2)
        d_zEvals     = (FrElement*)d_aux + N;                                         // [N..2N) rotated Z evaluations (round2)
        d_mapBuffers = (uint8_t*)d_aux + 2 * N * sizeof(FrElement);                   // [2N..2N+nConstraints*3*4) wire maps
        d_scanWork   = (FrElement*)d_aux + 4 * N;                                     // [4N..4N+scanWorkElems) scan block totals (within d_aux)

        // Static eval buffer slot aliases: S1(0), S2(1), S3(2), L1(3), QL(4), QR(5), QM(6), QO(7), QC(8)
        uint8_t* dStatic = (uint8_t*)d_staticEvalsBuffer;
        d_evalsS1 = dStatic + 0 * NExtBytes;
        d_evalsS2 = dStatic + 1 * NExtBytes;
        d_evalsS3 = dStatic + 2 * NExtBytes;
        d_evalsL1 = dStatic + 3 * NExtBytes;
        d_evalsQL = dStatic + 4 * NExtBytes;
        d_evalsQR = dStatic + 5 * NExtBytes;
        d_evalsQM = dStatic + 6 * NExtBytes;
        d_evalsQO = dStatic + 7 * NExtBytes;
        d_evalsQC = dStatic + 8 * NExtBytes;

        // Zkey coef aliases within d_staticEvalsBuffer slots 6 and 8
        FrElement* slot6 = (FrElement*)d_staticEvalsBuffer + 6 * NExt;
        d_coefQM = slot6;           
        d_coefQL = slot6 + N;
        d_coefQR = slot6 + 2*N;     
        d_coefQO = slot6 + 3*N;
        FrElement* slot8 = (FrElement*)d_staticEvalsBuffer + 8 * NExt;
        d_coefQC = slot8;           
        d_coefS1 = slot8 + N;
        d_coefS2 = slot8 + 2*N;     
        d_coefS3 = slot8 + 3*N;

        // T & T polynomial split aliases (reuse d_staticEvalsBuffer after evals are consumed)
        d_evalsT = d_piBuffer;             
        d_evalsTz = d_lagBuffer;
        d_t1 = d_staticEvalsBuffer;
        d_t2 = (FrElement*)d_staticEvalsBuffer + (N + 1);
        d_t3 = (FrElement*)d_staticEvalsBuffer + 2 * (N + 1);

        // Evals A, B, C aliases (reuse d_staticEvalsBuffer slots 4,5,7 after Q evals are consumed) 
        d_evalsA = d_evalsQL;
        d_evalsB = d_evalsQR;
        d_evalsC = d_evalsQO;

                   
        if (!pTauStream) pTauStream = gpu_plonk_create_cuda_stream_nonblocking();
        gpu_plonk_memcpy_h2d_async(d_ptau, PTau, ptauBytes, pTauStream);

        // Precompute omega power tables for all kernels (eliminates per-thread Fr::pow)
        if (!omegasStream) omegasStream = gpu_plonk_create_cuda_stream_nonblocking();
        FrElement omega_4x = fft->root(zkeyPower + 2, 1);
        FrElement omega = fft->root(zkeyPower, 1);
        gpu_plonk_precompute_omega_tables_async(d_omegaBasesNExt, d_omegaTidNExt, &omega_4x, 256, numBlocks4N, omegasStream);
        gpu_plonk_precompute_omega_tables_async(d_omegaBasesN, d_omegaTidN, &omega, 256, numBlocksN, omegasStream);

        if (!pinnedD2HStaging) {
            pinnedD2HStagingSize = (N + 6) * sizeof(FrElement);
            gpu_plonk_cuda_malloc_pinned_buffer(&pinnedD2HStaging, pinnedD2HStagingSize);
        }

        if (evalConstPols) {
            if (!evalNTTStream) evalNTTStream = gpu_plonk_create_cuda_stream_nonblocking();
            LOG_TRACE("··· Launching async eval NTT (9 polys from coefficients)");
            asyncEvalNTT = std::thread([this]() {
                void* slots[9] = {d_evalsS1, d_evalsS2, d_evalsS3, d_evalsL1,
                                    d_evalsQL, d_evalsQR, d_evalsQM, d_evalsQO, d_evalsQC};
                const char* names[9] = {"Sigma1", "Sigma2", "Sigma3", "L1",
                                          "QL", "QR", "QM", "QO", "QC"};
                for (int i = 0; i < 9; i++) {
                    gpu_plonk_memcpy_h2d_async(slots[i], polPtr[names[i]], NBytes, evalNTTStream);
                    gpu_plonk_zero_pad_async(slots[i], N, NExt, evalNTTStream);
                    gpu_plonk_sync_cuda_stream(evalNTTStream); 
                    ntt_bn128_gpu_dev_ptr(slots[i], zkeyPower + 2);
                }
                gpu_plonk_cuda_device_sync(); // ensure all sppark NTTs complete
            });
        } else {
            // Load pre-computed 4N evaluations from zkey file (async file I/O)
            void* dBuf = d_staticEvalsBuffer;

            FileReadFn readFn = [](void* dest, uint32_t sectionId, uint64_t offset, uint64_t len, void* ctx) {
                static_cast<BinFileUtils::BinFile*>(ctx)->readSectionToParallel(dest, sectionId, offset, len, 8);
            };

            std::array<uint32_t, 4> sSids = { Zkey::ZKEY_PL_SIGMA_SECTION, Zkey::ZKEY_PL_SIGMA_SECTION, Zkey::ZKEY_PL_SIGMA_SECTION, Zkey::ZKEY_PL_LAGRANGE_SECTION };
            std::array<uint64_t, 4> sOffs = { NBytes * 1, NBytes * 6, NBytes * 11, NBytes };
            std::array<uint64_t, 4> sSizes = { NExtBytes, NExtBytes, NExtBytes, NExtBytes };

            asyncTransferSigma = std::thread([readFn, dBuf, this,
                                              sSids, sOffs, sSizes]() {
                gpu_plonk_start_static_eval_transfer(readFn, (void*)this->fdZkeyPtr,
                                               dBuf, this->pinnedS, this->pinnedSize,
                                               sSids.data(), sOffs.data(), sSizes.data(), 4);
            });

            std::array<uint32_t, 5> qSids = {
                Zkey::ZKEY_PL_QL_SECTION, Zkey::ZKEY_PL_QR_SECTION, Zkey::ZKEY_PL_QM_SECTION,
                Zkey::ZKEY_PL_QO_SECTION, Zkey::ZKEY_PL_QC_SECTION };
            std::array<uint64_t, 5> qOffs = { NBytes, NBytes, NBytes, NBytes, NBytes};
            std::array<uint64_t, 5> qSizes = { NExtBytes, NExtBytes, NExtBytes, NExtBytes, NExtBytes };

            void* dBufQ = (void*)((uint8_t*)dBuf + 4 * NExtBytes);

            asyncTransferQ = std::thread([readFn, dBufQ, this,
                                          qSids, qOffs, qSizes]() {
                gpu_plonk_start_static_eval_transfer(readFn, (void*)this->fdZkeyPtr,
                                               dBufQ, this->pinnedQ, this->pinnedSize,
                                               qSids.data(), qOffs.data(), qSizes.data(), 5);
            });
        }

        LOG_TRACE("··· Uploading additions to GPU");
        uint32_t* signalId1 = new uint32_t[zkey->nAdditions];
        uint32_t* signalId2 = new uint32_t[zkey->nAdditions];
        FrElement* factor1 = new FrElement[zkey->nAdditions];
        FrElement* factor2 = new FrElement[zkey->nAdditions];
        for (uint32_t i = 0; i < zkey->nAdditions; i++) {
            signalId1[i] = additionsBuff[i].signalId1;
            signalId2[i] = additionsBuff[i].signalId2;
            factor1[i] = additionsBuff[i].factor1;
            factor2[i] = additionsBuff[i].factor2;
        }
        gpu_plonk_memcpy_h2d(d_addSignalId1, signalId1, zkey->nAdditions * sizeof(uint32_t));
        gpu_plonk_memcpy_h2d(d_addSignalId2, signalId2, zkey->nAdditions * sizeof(uint32_t));
        gpu_plonk_memcpy_h2d(d_addFactor1, factor1, zkey->nAdditions * sizeof(FrElement));
        gpu_plonk_memcpy_h2d(d_addFactor2, factor2, zkey->nAdditions * sizeof(FrElement));
        gpu_plonk_memcpy_h2d(d_additionLevels, additionLevels, zkey->nAdditions);
        delete[] signalId1;
        delete[] signalId2;
        delete[] factor1;
        delete[] factor2;

    }

    // ROUND 1
    template <typename Engine>
    void PlonkProverGPU<Engine>::round1()
    {
        // STEP 1.1 - Generate random blinding scalars (b_1, ..., b9) ∈ F

        // blindingFactors[0] unused — factors are 1-indexed (b_1 .. b_11)
        for (u_int32_t i = 1; i <= BLINDINGFACTORSLENGTH_PLONK_GPU; i++)
        {
            memset((void *)&(blindingFactors[i].v[0]), 0, sizeof(FrElement));
            randombytes_buf((void *)&(blindingFactors[i].v[0]), sizeof(FrElement) - 1);
        }

        // Upload blinding factors to device (persistent for all rounds)
        gpu_plonk_memcpy_h2d(d_blindings, blindingFactors, (BLINDINGFACTORSLENGTH_PLONK_GPU + 1) * sizeof(FrElement));

        // STEP 1.2 - Compute wire polynomials a(X), b(X) and c(X)
        LOG_TRACE("> Computing A, B, C wire polynomials");
#ifdef PLONK_GPU_TIMING
        double t0 = omp_get_wtime();
#endif
        computeWirePolynomials();
        gpu_plonk_cuda_device_sync(); // ensure we can call sppark functions that use their own stream
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]   wirePolynomials (3xIFFT+D2D): " << omp_get_wtime() - t0 << "s" << std::endl;
#endif

        // STEP 1.3 - Compute [a]_1, [b]_1, [c]_1
        // GPU-resident MSM — no D2H needed (A/B/C coefficients stay on GPU only)
        LOG_TRACE("> Computing A, B, C commitments (MSM devptr)");

        void* dSlots[3] = {d_polCoefA, d_polCoefB, d_polCoefC};
        std::string names[3] = {"A", "B", "C"};
        FrElement bfs[3][2] = {
            {blindingFactors[2], blindingFactors[1]},
            {blindingFactors[4], blindingFactors[3]},
            {blindingFactors[6], blindingFactors[5]},
        };

        // Ensure async PTau H2D (launched in round0/preAllocate) is complete before first MSM
        gpu_plonk_sync_cuda_stream(pTauStream);

        for (int w = 0; w < 3; w++) {
#ifdef PLONK_GPU_TIMING
            double tMsm = omp_get_wtime();
#endif
            G1Point pt = multiExponentiationGPU_devptr(dSlots[w], N);
            applyBlindingCorrection(pt, bfs[w], 2);
            proof->addPolynomialCommitment(names[w], pt);
#ifdef PLONK_GPU_TIMING
            std::cout << "[TIMING]   MSM " << names[w] << " (devptr): " << omp_get_wtime() - tMsm << "s" << std::endl;
#endif
        }
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::computeWirePolynomials()
    {

        uint32_t nDirect = zkey->nVars - zkey->nAdditions;
        size_t mapBytes = zkey->nConstraints * 3 * sizeof(uint32_t);

        if (!evalConstPols) {
            // Fill buffers["publics"][0..nPublic) on CPU for transcript/PI
            for (uint32_t i = 0; i < zkey->nPublic; i++) {
                FrElement w = getWitness(mapBuffers["A"][i]);
                E.fr.toMontgomery(buffers["publics"][i], w);
            }

            // Launch async PI computation on GPU (file-based L_j loading)
            FileReadFn piReadFn = [](void* dest, uint32_t sectionId,
                                        uint64_t offset, uint64_t len, void* ctx) {
                static_cast<BinFileUtils::BinFile*>(ctx)->readSectionToParallel(
                    dest, sectionId, offset, len, 8);
            };

            asyncComputePI = std::thread([piReadFn, this]() {
                gpu_plonk_compute_pi(nullptr, piReadFn, (void*)this->fdZkeyPtr,
                                Zkey::ZKEY_PL_LAGRANGE_SECTION,
                                this->NBytes, this->NBytes * 5,
                                (const void*)this->buffers["publics"], this->NExt, this->zkey->nPublic,
                                this->d_piBuffer, this->d_lagBuffer, this->pinnedPI, this->pinnedSize);
            });
        }

        LOG_TRACE("··· H2D maps to GPU");
        gpu_plonk_memcpy_h2d(d_mapBuffers, mapBuffersBigBuffer, mapBytes);

        // GPU gather + IFFT for each wire polynomial
        void* dPolSlots[3] = {d_polCoefA, d_polCoefB, d_polCoefC};
        std::string wireNames[3] = {"A", "B", "C"};
        uint32_t* dMap = (uint32_t*)d_mapBuffers;

        for (int w = 0; w < 3; w++) {
            std::ostringstream ss;
            ss << "··· Computing " << wireNames[w] << " gather+Montgomery+IFFT (GPU)";
            LOG_TRACE(ss);

            // GPU gather: reads witness+map from d_aux[2N..], writes to d_gathered[0..N)
            gpu_plonk_gather_witness(d_gathered, dMap + w * zkey->nConstraints,
                               d_witness, d_intWitness,
                               nDirect, zkey->nConstraints, N);
            gpu_plonk_cuda_device_sync(); // synch before sspark

            // IFFT in-place on d_gathered (sppark uses own non-blocking stream)
            intt_bn128_gpu_dev_ptr(d_gathered, zkeyPower);

            // D2D to persistent GPU slot
            gpu_plonk_memcpy_d2d(dPolSlots[w], d_gathered, N * sizeof(FrElement));
        }
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
            transcript->addScalar(buffers["publics"][i]);
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
#ifdef PLONK_GPU_TIMING
        double t0 = omp_get_wtime();
#endif
        computeZ();
        gpu_plonk_cuda_device_sync(); // ensure we can call sppark functions that use their own stream
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]   computeZ (ratios+product+IFFT): " << omp_get_wtime() - t0 << "s" << std::endl;
#endif
    
        LOG_TRACE("> Computing Z polynomial commitment (MSM devptr)");
#ifdef PLONK_GPU_TIMING
        t0 = omp_get_wtime();
#endif
        G1Point Z = multiExponentiationGPU_devptr(d_polCoefZ, N);
        FrElement bfZ[3] = {blindingFactors[9], blindingFactors[8], blindingFactors[7]};
        applyBlindingCorrection(Z, bfZ, 3);
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]   MSM Z (devptr): " << omp_get_wtime() - t0 << "s" << std::endl;
#endif
        proof->addPolynomialCommitment("Z", Z);
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::computeZ()
    {
        // Wait for sigma transfer to complete (S1,S2,S3 needed by z_ratios kernel)
        if ( !evalConstPols && asyncTransferSigma.joinable()) {
#ifdef PLONK_GPU_TIMING
            double t0 = omp_get_wtime();
#endif
            asyncTransferSigma.join();
#ifdef PLONK_GPU_TIMING
            std::cout<<"[computeZ] sigma join completed, waited: " << omp_get_wtime() - t0 <<endl;
#endif
        }

        LOG_TRACE("··· Computing Z evaluations (GPU prefix scan)");

        auto beta = challenges["beta"];
        auto gamma = challenges["gamma"];
        auto k1 = *((FrElement *)zkey->k1);
        auto k2 = *((FrElement *)zkey->k2);

        uint32_t nDirect = zkey->nVars - zkey->nAdditions;
        uint32_t* dMap = (uint32_t*)d_mapBuffers;

        // Step 1: Fused gather+z_ratios on GPU (omega tables precomputed in round0)
#ifdef PLONK_GPU_TIMING
        double tz0 = omp_get_wtime();
#endif
        // Ensure async omega table precomputation (launched in round0/preAllocate) is complete before z_ratios kernel
        gpu_plonk_sync_cuda_stream(omegasStream);

        gpu_plonk_compute_z_ratios_gather(
            d_ratios,
            dMap, dMap + zkey->nConstraints, dMap + 2 * zkey->nConstraints,
            d_witness, d_intWitness, nDirect, zkey->nConstraints,
            d_staticEvalsBuffer,
            (const void*)&beta, (const void*)&gamma,
            (const void*)&k1, (const void*)&k2,
            N, d_omegaBasesN, d_omegaTidN);
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]     z_ratios GPU (gather): " << omp_get_wtime() - tz0 << "s" << std::endl;
#endif

        // Step 2: GPU inclusive multiplicative prefix scan on d_ratios[0..N)
#ifdef PLONK_GPU_TIMING
        tz0 = omp_get_wtime();
#endif
        gpu_plonk_prefix_scan_multiply(d_ratios, N, d_scanWork);

        // Step 3: Rotate left by 1: d_ratios[0..N) → d_zEvals[0..N)
        // Z[0] = scan[N-1] (total product), Z[i] = scan[i-1] for i >= 1
        gpu_plonk_rotate_left(d_zEvals, d_ratios, N);
        gpu_plonk_cuda_device_sync(); // ensure we can call sppark functions that use their own stream
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]     z GPU prefix scan + rotate: " << omp_get_wtime() - tz0 << "s" << std::endl;
#endif

        // Step 4: IFFT on d_zEvals — evaluations already on GPU
        LOG_TRACE("··· Computing Z ifft (GPU dev_ptr)");
#ifdef PLONK_GPU_TIMING
        tz0 = omp_get_wtime();
#endif
        intt_bn128_gpu_dev_ptr(d_zEvals, zkeyPower);

        // Step 6: D2D to persistent d_polCoefZ slot
        gpu_plonk_memcpy_d2d(d_polCoefZ, d_zEvals, N * sizeof(FrElement));
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]     z IFFT+D2D: " << omp_get_wtime() - tz0 << "s" << std::endl;
#endif
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
#ifdef PLONK_GPU_TIMING
        double t0 = omp_get_wtime();
#endif
        computeT();
        gpu_plonk_cuda_device_sync(); // computeT kernels (default stream) → sppark MSMs
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]   computeT total: " << omp_get_wtime() - t0 << "s" << std::endl;
#endif

        // Qcoef transfer batch 2: QM, QL, QR, QO → slot 6 (QM eval slot, now free)
        // Batch 1 (QC, S1, S2, S3 → slot 8) was launched in computeT after gate_C.        
        size_t polyBytes = N * sizeof(FrElement);

        std::array<void*, 4> dsts = {d_coefQM, d_coefQL, d_coefQR, d_coefQO};
        std::array<const void*, 4> srcs = {polPtr["QM"], polPtr["QL"],polPtr["QR"], polPtr["QO"]};
        std::array<size_t, 4> sizes = {polyBytes, polyBytes, polyBytes, polyBytes};

        asyncTransferPolsBatch2 = std::thread([dsts, srcs, sizes, this]() {
            void* d[4]; 
            const void* s[4]; 
            size_t sz[4];
            for (int i = 0; i < 4; i++) { d[i] = dsts[i]; s[i] = srcs[i]; sz[i] = sizes[i]; }
            gpu_plonk_start_cpu_to_gpu_transfer(d, s, sz, 4, this->pinnedQ, this->pinnedSize);
        });
    
        LOG_TRACE("> Computing T1, T2 & T3 commitments (MSM devptr)");

        void* dTSlots[3] = {d_t1, d_t2, d_t3};
        std::string tNames[3] = {"T1", "T2", "T3"};
        size_t tNpoints[3] = {N+1, N+1, N+6};

        for (int t = 0; t < 3; t++) {
#ifdef PLONK_GPU_TIMING
            double tMsm = omp_get_wtime();
#endif
            G1Point pt = multiExponentiationGPU_devptr(dTSlots[t], tNpoints[t]);
            proof->addPolynomialCommitment(tNames[t], pt);
#ifdef PLONK_GPU_TIMING
            std::cout << "[TIMING]   MSM " << tNames[t] << " (devptr): " << omp_get_wtime() - tMsm << "s" << std::endl;
#endif
        }
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::computeT()
    {
        LOG_TRACE("··· Computing T evaluations (incremental GPU)");

        // Preloaded constants
        auto beta = challenges["beta"];
        auto gamma = challenges["gamma"];
        auto alpha = challenges["alpha"];
        auto alpha2 = challenges["alpha2"];
        auto k1 = *((FrElement *)zkey->k1);
        auto k2 = *((FrElement *)zkey->k2);
        auto omega1 = fft->root(zkeyPower, 1);
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

        // Upload Z1/Z2/Z3 to d_zvals on GPU (used by QM+perm+L1 kernel via shared memory)
        gpu_plonk_memcpy_h2d(d_zvals, Z1, 4 * sizeof(FrElement));
        gpu_plonk_memcpy_h2d((uint8_t*)d_zvals + 4 * sizeof(FrElement), Z2, 4 * sizeof(FrElement));
        gpu_plonk_memcpy_h2d((uint8_t*)d_zvals + 8 * sizeof(FrElement), Z3, 4 * sizeof(FrElement));

        // Wait for async PI computation (launched in computeWirePolynomials, only when !evalConstPols)
        if (!evalConstPols && asyncComputePI.joinable()) {
#ifdef PLONK_GPU_TIMING
            double t0 = omp_get_wtime();
#endif
            asyncComputePI.join();
#ifdef PLONK_GPU_TIMING
            std::cout << "[computeT] PI join waited: " << omp_get_wtime() - t0 << endl;
#endif
        }
        // Wait for async Q eval transfer to complete (QL-QC needed by gate kernels)
        if (!evalConstPols && asyncTransferQ.joinable()) {
#ifdef PLONK_GPU_TIMING
            double t0 = omp_get_wtime();
#endif
            asyncTransferQ.join();
#ifdef PLONK_GPU_TIMING
            std::cout << " [computeT] Q join waited: " << omp_get_wtime() - t0 << endl;
#endif
        }

#ifdef PLONK_GPU_TIMING
        double tstart = omp_get_wtime();
        double tWire;
#endif

        // --- Wire A: D2D unblinded coefs → zero-pad → FFT → gate_A → D2D to evalsA slot ---
        LOG_TRACE("··· Computing A evals + gate_A (GPU)");
#ifdef PLONK_GPU_TIMING
        tWire = omp_get_wtime();
#endif
        gpu_plonk_memcpy_d2d(d_aux, d_polCoefA, N * sizeof(FrElement));
        gpu_plonk_zero_pad(d_aux, N, NExt);
        gpu_plonk_cuda_device_sync(); // default-stream zero_pad → sppark NTT
        ntt_bn128_gpu_dev_ptr(d_aux, zkeyPower + 2);
        gpu_plonk_compute_gate_a(d_evalsT, d_evalsTz, d_aux, d_evalsQL,
                           d_blindings, N,
                           d_omegaBasesNExt, d_omegaTidNExt);
        gpu_plonk_memcpy_d2d(d_evalsA, d_aux, NExtBytes);
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]     wire A (D2D+pad+NTT+gate+D2D): " << omp_get_wtime() - tWire << "s" << std::endl;
#endif

        // --- Wire B: D2D → zero-pad → FFT → gate_B → D2D to evalsB slot ---
        LOG_TRACE("··· Computing B evals + gate_B (GPU)");
#ifdef PLONK_GPU_TIMING
        tWire = omp_get_wtime();
#endif
        gpu_plonk_memcpy_d2d(d_aux, d_polCoefB, N * sizeof(FrElement));
        gpu_plonk_zero_pad(d_aux, N, NExt);
        gpu_plonk_cuda_device_sync(); // default-stream zero_pad → sppark NTT
        ntt_bn128_gpu_dev_ptr(d_aux, zkeyPower + 2);
        gpu_plonk_compute_gate_b(d_evalsT, d_evalsTz, d_aux, d_evalsQR,
                           d_blindings, N,
                           d_omegaBasesNExt, d_omegaTidNExt);
        gpu_plonk_memcpy_d2d(d_evalsB, d_aux, NExtBytes);
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]     wire B (D2D+pad+NTT+gate+D2D): " << omp_get_wtime() - tWire << "s" << std::endl;
#endif

        // --- Wire C: D2D → zero-pad → FFT → gate_C → D2D to evalsC slot ---
        LOG_TRACE("··· Computing C evals + gate_C (GPU)");
#ifdef PLONK_GPU_TIMING
        tWire = omp_get_wtime();
#endif
        gpu_plonk_memcpy_d2d(d_aux, d_polCoefC, N * sizeof(FrElement));
        gpu_plonk_zero_pad(d_aux, N, NExt);
        gpu_plonk_cuda_device_sync(); // default-stream zero_pad → sppark NTT
        ntt_bn128_gpu_dev_ptr(d_aux, zkeyPower + 2);
        gpu_plonk_compute_gate_c(d_evalsT, d_evalsTz, d_aux, d_evalsQO, d_evalsQC,
                           d_blindings, N,
                           d_omegaBasesNExt, d_omegaTidNExt);
        gpu_plonk_memcpy_d2d(d_evalsC, d_aux, NExtBytes);
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]     wire C (D2D+pad+NTT+gate+D2D): " << omp_get_wtime() - tWire << "s" << std::endl;
#endif

        // --- Early zkey coef transfer batch 1: QC, S1, S2, S3 → slot 8 ---
        // Slot 8 (QC evals) is now free — gate_C was the last reader.         
        gpu_plonk_cuda_device_sync(); // Sync to ensure gate_C kernel can be used
        
        std::array<void*, 4> dsts = {d_coefQC, d_coefS1, d_coefS2, d_coefS3};
        std::array<const void*, 4> srcs = { polPtr["QC"], polPtr["Sigma1"], polPtr["Sigma2"], polPtr["Sigma3"]};
        std::array<size_t, 4> sizes = {NBytes, NBytes, NBytes, NBytes};

        asyncTransferPolsBatch1 = std::thread([dsts, srcs, sizes, this]() {
            void* d[4]; const void* s[4]; size_t sz[4];
            for (int i = 0; i < 4; i++) { d[i] = dsts[i]; s[i] = srcs[i]; sz[i] = sizes[i]; }
            gpu_plonk_start_cpu_to_gpu_transfer(d, s, sz, 4, this->pinnedS, this->pinnedSize);
        });
        
        // --- Z: D2D → zero-pad → FFT → set wrap-around ---
        LOG_TRACE("··· Computing Z evals (GPU)");
#ifdef PLONK_GPU_TIMING
        tWire = omp_get_wtime();
#endif
        gpu_plonk_memcpy_d2d(d_aux, d_polCoefZ, N * sizeof(FrElement));
        gpu_plonk_zero_pad(d_aux, N, NExt);
        gpu_plonk_cuda_device_sync(); // default-stream zero_pad → sppark NTT
        ntt_bn128_gpu_dev_ptr(d_aux, zkeyPower + 2);
        // Wrap-around: copy first 4 elements to d_aux[NExt..NExt+4) so QM+perm+L1 kernel can read z(x*omega)
        gpu_plonk_memcpy_d2d((uint8_t*)d_aux + NExtBytes, d_aux, 4 * sizeof(FrElement));
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]     wire Z (D2D+pad+NTT+wrap): " << omp_get_wtime() - tWire << "s" << std::endl;
#endif

        // --- QM + Permutation + L1 kernel --- (rest of constraints)
#ifdef PLONK_GPU_TIMING
        tWire = omp_get_wtime();
#endif
        LOG_TRACE("··· Computing QM + permutation + L1 (GPU)");
        gpu_plonk_compute_qm_perm_l1(
            d_evalsT, d_evalsTz,
            d_evalsA, d_evalsB, d_evalsC,
            d_aux,
            d_evalsQM, d_evalsS1, d_evalsS2, d_evalsS3, d_evalsL1,
            d_blindings, d_zvals,
            (const void*)&beta, (const void*)&gamma,
            (const void*)&alpha, (const void*)&alpha2,
            (const void*)&k1, (const void*)&k2,
            (const void*)&omega1,
            N,
            d_omegaBasesNExt, d_omegaTidNExt);
        
            gpu_plonk_cuda_device_sync(); // ensure qm_perm_l1 is done before we can call sppark functions that use their own stream

#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]     QM+perm+L1 kernel: " << omp_get_wtime() - tWire << "s" << std::endl;
        std::cout << "[TIMING]     incremental T evals total: " << omp_get_wtime() - tstart << "s" << std::endl;
#endif

        // T IFFT in-place
        LOG_TRACE("··· Computing T ifft (GPU)");
#ifdef PLONK_GPU_TIMING
        tWire = omp_get_wtime();
#endif
        intt_bn128_gpu_dev_ptr(d_evalsT, zkeyPower + 2);
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]     T INTT (GPU): " << omp_get_wtime() - tWire << "s" << std::endl;
#endif

        // Tz IFFT in-place 
        LOG_TRACE("··· Computing Tz ifft (GPU)");
#ifdef PLONK_GPU_TIMING
        tWire = omp_get_wtime();
#endif
        intt_bn128_gpu_dev_ptr(d_evalsTz, zkeyPower + 2);
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]     Tz INTT (GPU): " << omp_get_wtime() - tWire << "s" << std::endl;
#endif

        // GPU: divZh + T+Tz add in one fused kernel
#ifdef PLONK_GPU_TIMING
        tWire = omp_get_wtime();
#endif
        gpu_plonk_divzh_add(d_evalsT, d_evalsTz, N);
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]     divZh+add (GPU): " << omp_get_wtime() - tWire << "s" << std::endl;
#endif
    
        // GPU: Split T into T1/T2/T3 with blinding,
        LOG_TRACE("··· Computing T1, T2, T3 polynomials (GPU split+blind)");
#ifdef PLONK_GPU_TIMING
        tWire = omp_get_wtime();
#endif

        gpu_plonk_split_t_blinding(d_t1, d_t2, d_t3, d_evalsT,
                             d_blindings, N);

        // D2H deferred to round3 where it overlaps with MSMs via async stream
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]     T split+blind (GPU): " << omp_get_wtime() - tWire << "s" << std::endl;
#endif
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

        // Ensure zkey H2D batch 1 (QC,S1,S2,S3 → slot 8) is complete — needed for S1/S2 evals below
        if (asyncTransferPolsBatch1.joinable()) asyncTransferPolsBatch1.join();
        
        auto xi = challenges["xi"];
        auto xiw = challenges["xiw"];

        // Precompute xi^N for blinding corrections 
        FrElement xiN;
        E.fr.copy(xiN, xi);
        for (uint32_t p = 0; p < zkeyPower; p++) E.fr.square(xiN, xiN);

        // Helper: compute blinding correction = sum_j bf[j] * (x^(N+j) - x^j)
        auto blindCorr = [&](FrElement &result, const FrElement bf[], int len,
                             const FrElement &x, const FrElement &xN) {
            FrElement xj = E.fr.one();
            FrElement xNj = xN;
            for (int j = 0; j < len; j++) {
                FrElement diff, corr;
                E.fr.sub(diff, xNj, xj);
                E.fr.mul(corr, bf[j], diff);
                E.fr.add(result, result, corr);
                E.fr.mul(xj, xj, x);
                E.fr.mul(xNj, xNj, x);
            }
        };

        // GPU evaluations: A(xi), B(xi), C(xi) — unblinded GPU coefs + blinding correction
        FrElement eval_a, eval_b, eval_c;
        gpu_plonk_poly_eval_to_host(&eval_a, d_polCoefA, &xi, N, d_aux);
        gpu_plonk_poly_eval_to_host(&eval_b, d_polCoefB, &xi, N, d_aux);
        gpu_plonk_poly_eval_to_host(&eval_c, d_polCoefC, &xi, N, d_aux);

        FrElement bfA[2] = {blindingFactors[2], blindingFactors[1]};
        FrElement bfB[2] = {blindingFactors[4], blindingFactors[3]};
        FrElement bfC[2] = {blindingFactors[6], blindingFactors[5]};
        blindCorr(eval_a, bfA, 2, xi, xiN);
        blindCorr(eval_b, bfB, 2, xi, xiN);
        blindCorr(eval_c, bfC, 2, xi, xiN);

        // GPU evaluations: Sigma1(xi), Sigma2(xi) — from d_coefS1/S2 (no blinding)
        FrElement eval_s1, eval_s2;
        gpu_plonk_poly_eval_to_host(&eval_s1, d_coefS1, &xi, N, d_aux);
        gpu_plonk_poly_eval_to_host(&eval_s2, d_coefS2, &xi, N, d_aux);

        // GPU evaluation: Z(xiw) — unblinded GPU coefs + blinding correction
        FrElement eval_zw;
        gpu_plonk_poly_eval_to_host(&eval_zw, d_polCoefZ, &xiw, N, d_aux);
        FrElement bfZ[3] = {blindingFactors[9], blindingFactors[8], blindingFactors[7]};
        blindCorr(eval_zw, bfZ, 3, xiw, xiN);

        proof->addEvaluationCommitment("eval_a", eval_a);
        proof->addEvaluationCommitment("eval_b", eval_b);
        proof->addEvaluationCommitment("eval_c", eval_c);
        proof->addEvaluationCommitment("eval_s1", eval_s1);
        proof->addEvaluationCommitment("eval_s2", eval_s2);
        proof->addEvaluationCommitment("eval_zw", eval_zw);
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
        // Ensure zkey H2D batch 2 (QM,QL,QR,QO → slot 6) is complete — needed by computeR kernel
        if (asyncTransferPolsBatch2.joinable()) asyncTransferPolsBatch2.join();

        LOG_TRACE("> Computing linearisation polynomial R(X)");
#ifdef PLONK_GPU_TIMING
        double t0 = omp_get_wtime();
#endif
        computeR();
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]   computeR+Wxi_numerator (GPU): " << omp_get_wtime() - t0 << "s" << std::endl;
#endif

        // STEP 5.3 Compute opening proof polynomial Wxi(X)
        // After computeWxi, result is in d_aux on GPU — MSM directly from there
        LOG_TRACE("> Computing opening proof polynomial Wxi(X) polynomial");
#ifdef PLONK_GPU_TIMING
        t0 = omp_get_wtime();
#endif
        computeWxi();
        gpu_plonk_cuda_device_sync(); // divByZerofier (default stream) → sppark MSM
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]   computeWxi (divByZerofier): " << omp_get_wtime() - t0 << "s" << std::endl;
#endif

        LOG_TRACE("> Computing Wxi polynomial commitment (GPU devptr)");
#ifdef PLONK_GPU_TIMING
        t0 = omp_get_wtime();
#endif
        // Wxi is in d_aux after divByZerofier, use conservative npoints (upper bound on degree)
        G1Point commitWxi = multiExponentiationGPU_devptr(d_aux, N + 5);
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]   MSM Wxi (devptr): " << omp_get_wtime() - t0 << "s" << std::endl;
#endif

        // STEP 5.4 Compute opening proof polynomial Wxiw(X)
        LOG_TRACE("> Computing opening proof polynomial Wxiw(X) polynomial");
#ifdef PLONK_GPU_TIMING
        t0 = omp_get_wtime();
#endif
        computeWxiw();
        gpu_plonk_cuda_device_sync(); // divByZerofier (default stream) → sppark MSM
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]   computeWxiw: " << omp_get_wtime() - t0 << "s" << std::endl;
#endif

        LOG_TRACE("> Computing Wxiw polynomial commitment (GPU devptr)");
#ifdef PLONK_GPU_TIMING
        t0 = omp_get_wtime();
#endif
        // After divByZerofier: blinded Z (N+3) → degree reduced by 1 → N+2 coefficients
        G1Point commitWxiw = multiExponentiationGPU_devptr(d_aux, N + 2);
#ifdef PLONK_GPU_TIMING
        std::cout << "[TIMING]   MSM Wxiw (devptr): " << omp_get_wtime() - t0 << "s" << std::endl;
#endif

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

        auto n = E.fr.set(N);
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

        // Precompute r0
        auto r0 = E.fr.sub(eval_pi, E.fr.mul(e3, E.fr.add(proof->getEvaluationCommitment("eval_c"), challenges["gamma"])));
        r0 = E.fr.sub(r0, e4);

        ss.str("");
        ss << "··· r0: " << E.fr.toString(r0);
        LOG_TRACE(ss);

        // Precompute combined scalars for fused R+Wxi loop
        auto e2_plus_e4 = E.fr.add(e2, e4);
        auto e3_beta = E.fr.mul(e3, challenges["beta"]);
        FrElement neg_zh;
        E.fr.neg(neg_zh, challenges["zh"]);
        auto xin = challenges["xin"];
        auto xin2 = E.fr.square(xin);

        auto eval_a = proof->getEvaluationCommitment("eval_a");
        auto eval_b = proof->getEvaluationCommitment("eval_b");
        auto eval_c = proof->getEvaluationCommitment("eval_c");
        auto v1 = challenges["v1"];
        auto v2 = challenges["v2"];
        auto v3 = challenges["v3"];
        auto v4 = challenges["v4"];
        auto v5 = challenges["v5"];
        auto eval_s1 = proof->getEvaluationCommitment("eval_s1");
        auto eval_s2 = proof->getEvaluationCommitment("eval_s2");

        auto wxi_offset = E.fr.mul(v1, eval_a);
        wxi_offset = E.fr.add(wxi_offset, E.fr.mul(v2, eval_b));
        wxi_offset = E.fr.add(wxi_offset, E.fr.mul(v3, eval_c));
        wxi_offset = E.fr.add(wxi_offset, E.fr.mul(v4, eval_s1));
        wxi_offset = E.fr.add(wxi_offset, E.fr.mul(v5, eval_s2));

        // GPU R+Wxi kernel: writes Wxi directly to d_aux on GPU
        struct {
            FrElement coef_ab, eval_a, eval_b, eval_c;
            FrElement e2_plus_e4, e3_beta;
            FrElement v1, v2, v3, v4, v5;
            FrElement neg_zh, xin, xin2;
            FrElement r0, wxi_offset;
            FrElement blindDelta[3];
            uint64_t N;
        } rwxiConst;
        rwxiConst.coef_ab = coef_ab;
        rwxiConst.eval_a = eval_a;
        rwxiConst.eval_b = eval_b;
        rwxiConst.eval_c = eval_c;
        rwxiConst.e2_plus_e4 = e2_plus_e4;
        rwxiConst.e3_beta = e3_beta;
        rwxiConst.v1 = v1;
        rwxiConst.v2 = v2;
        rwxiConst.v3 = v3;
        rwxiConst.v4 = v4;
        rwxiConst.v5 = v5;
        rwxiConst.neg_zh = neg_zh;
        rwxiConst.xin = xin;
        rwxiConst.xin2 = xin2;
        rwxiConst.r0 = r0;
        rwxiConst.wxi_offset = wxi_offset;

        // Blinding corrections:
        // blindDelta[j] = v1*bfA[j] + v2*bfB[j] + v3*bfC[j] + e2_plus_e4*bfZ[j]
        // where bfA = {bf[2],bf[1]}, bfB = {bf[4],bf[3]}, bfC = {bf[6],bf[5]}, bfZ = {bf[9],bf[8],bf[7]}
        rwxiConst.blindDelta[0] = E.fr.add(
            E.fr.add(E.fr.mul(v1, blindingFactors[2]), E.fr.mul(v2, blindingFactors[4])),
            E.fr.add(E.fr.mul(v3, blindingFactors[6]), E.fr.mul(e2_plus_e4, blindingFactors[9])));
        rwxiConst.blindDelta[1] = E.fr.add(
            E.fr.add(E.fr.mul(v1, blindingFactors[1]), E.fr.mul(v2, blindingFactors[3])),
            E.fr.add(E.fr.mul(v3, blindingFactors[5]), E.fr.mul(e2_plus_e4, blindingFactors[8])));
        rwxiConst.blindDelta[2] = E.fr.mul(e2_plus_e4, blindingFactors[7]);

        rwxiConst.N = N;

        gpu_plonk_compute_r_wxi(
            d_aux,
            d_polCoefA, d_polCoefB, d_polCoefC, d_polCoefZ,
            d_coefQM, d_coefQL, d_coefQR, d_coefQO, d_coefQC,
            d_coefS1, d_coefS2, d_coefS3,
            d_t1, d_t2, d_t3,
            (const void*)&rwxiConst, N);

        // Wxi result is now in d_aux[0..N+5] on GPU
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::computeWxi()
    {
        // Wxi numerator already computed by GPU kernel in computeR() 
        uint64_t len = N + 6;
        FrElement xi = challenges["xi"];
        FrElement invBeta;
        E.fr.inv(invBeta, xi);         
        FrElement invBetaNeg;
        E.fr.neg(invBetaNeg, invBeta);

        // Get Wxi[0] from GPU 
        FrElement wxi0;
        gpu_plonk_memcpy_d2h(&wxi0, d_aux, sizeof(FrElement));
        FrElement y0 = E.fr.mul(invBetaNeg, wxi0);

        // GPU affine scan — workspace starts after coefficient data
        void* dPairWork = (void*)((FrElement*)d_aux + len);
        gpu_plonk_compute_div_zerofier(d_aux, len, (const void*)&invBeta, (const void*)&y0, dPairWork);
        // Result stays in d_aux for GPU-direct MSM 
    }

    template <typename Engine>
    void PlonkProverGPU<Engine>::computeWxiw()
    {
        uint64_t len = N + 3;

        // D2D unblinded Z coefficients → d_aux[0..N)
        gpu_plonk_memcpy_d2d(d_aux, d_polCoefZ, N * sizeof(FrElement));

        // Apply blinding corrections on CPU:
        FrElement bf[3] = {blindingFactors[9], blindingFactors[8], blindingFactors[7]};
        FrElement eval_zw = proof->getEvaluationCommitment("eval_zw");

        // D2H low coefficients [0..3) from GPU
        FrElement lowCoef[3];
        gpu_plonk_memcpy_d2h(lowCoef, d_aux, 3 * sizeof(FrElement));

        // Apply low blind corrections + eval_zw subtraction
        lowCoef[0] = E.fr.sub(E.fr.sub(lowCoef[0], bf[0]), eval_zw);
        lowCoef[1] = E.fr.sub(lowCoef[1], bf[1]);
        lowCoef[2] = E.fr.sub(lowCoef[2], bf[2]);

        // H2D corrected low coefficients + high blinding terms
        gpu_plonk_memcpy_h2d(d_aux, lowCoef, 3 * sizeof(FrElement));
        gpu_plonk_memcpy_h2d((FrElement*)d_aux + N, bf, 3 * sizeof(FrElement));

        // Compute divByZerofier parameters: invBeta = 1/xiw, y0 = invBetaNeg * coef[0]
        FrElement xiw = challenges["xiw"];
        FrElement invBeta;
        E.fr.inv(invBeta, xiw);
        FrElement invBetaNeg;
        E.fr.neg(invBetaNeg, invBeta);
        FrElement y0 = E.fr.mul(invBetaNeg, lowCoef[0]);

        // GPU divByZerofier
        void* dPairWork = (void*)((FrElement*)d_aux + len);
        gpu_plonk_compute_div_zerofier(d_aux, len, (const void*)&invBeta, (const void*)&y0, dPairWork);
        // Result stays in d_aux for GPU-direct MSM (no D2H needed)
    }

    // CPU correction for unblinded MSM: adds bf[i]*(PTau[N+i] - PTau[i]) for each blinding factor.
    template <typename Engine>
    void PlonkProverGPU<Engine>::applyBlindingCorrection(G1Point &commitment, FrElement *bFactors, u_int32_t nFactors)
    {
        for (u_int32_t i = 0; i < nFactors; i++) {
            FrElement scalar;
            G1Point ptN, pt0, diff, term;
            E.fr.fromMontgomery(scalar, bFactors[i]);
            E.g1.copy(ptN, PTau[N + i]);
            E.g1.copy(pt0, PTau[i]);
            E.g1.sub(diff, ptN, pt0);
            E.g1.mulByScalar(term, diff, (uint8_t *)&scalar, sizeof(scalar));
            E.g1.add(commitment, commitment, term);
        }
    }

    // GPU MSM from device-resident Montgomery-form scalars (mont=true)
    template <typename Engine>
    typename Engine::G1Point PlonkProverGPU<Engine>::multiExponentiationGPU_devptr(
        void* dScalars, size_t npoints)
    {
        G1Point value;

        struct JacobianPoint {
            typename Engine::F1Element X;
            typename Engine::F1Element Y;
            typename Engine::F1Element Z;
        };
        JacobianPoint gpuResult;

        msm_bn128_gpu_dev_ptr(&gpuResult, d_ptau, dScalars, npoints, true);

        // Convert from standard Jacobian to Extended Jacobian
        value.x = gpuResult.X;
        value.y = gpuResult.Y;
        E.f1.square(value.zz, gpuResult.Z);
        E.f1.mul(value.zzz, value.zz, gpuResult.Z);

        return value;
    }

}