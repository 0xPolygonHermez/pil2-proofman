#include "plonk_prover.hpp"

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

namespace Plonk
{
    template <typename Engine>
    void PlonkProver<Engine>::initialize(void *reservedMemoryPtr, uint64_t reservedMemorySize)
    {
        zkey = NULL;
        this->reservedMemoryPtr = (FrElement *)reservedMemoryPtr;
        this->reservedMemorySize = reservedMemorySize;

        curveName = CurveUtils::getCurveNameByEngine();
    }

    template <typename Engine>
    PlonkProver<Engine>::PlonkProver(Engine &_E) : E(_E)
    {
        initialize(NULL);
    }

    template <typename Engine>
    PlonkProver<Engine>::PlonkProver(Engine &_E, void *reservedMemoryPtr, uint64_t reservedMemorySize) : E(_E)
    {
        initialize(reservedMemoryPtr, reservedMemorySize);
    }

    template <typename Engine>
    PlonkProver<Engine>::~PlonkProver()
    {
        this->removePrecomputedData();

        delete transcript;
        delete proof;
    }

    template <typename Engine>
    void PlonkProver<Engine>::removePrecomputedData()
    {
        // DELETE RESERVED MEMORY (if necessary)
        delete[] precomputedBigBuffer;
        delete[] mapBuffersBigBuffer;
        delete[] buffInternalWitness;

        if (NULL == reservedMemoryPtr)
        {
            delete[] inverses;
            delete[] products;
            delete[] nonPrecomputedBigBuffer;
        }

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
        cout << 1 << endl;

        delete evaluations["QL"];
        delete evaluations["QR"];
        delete evaluations["QM"];
        delete evaluations["QO"];
        delete evaluations["QC"];
        delete evaluations["Sigma1"];
        delete evaluations["Sigma2"];
        delete evaluations["Sigma3"];
        delete evaluations["lagrange"];
        cout << 1 << endl;
    }

    template <typename Engine>
    void PlonkProver<Engine>::setZkey(BinFileUtils::BinFile *fdZkey)
    {
        // try
        // {
        if (NULL != zkey)
        {
            removePrecomputedData();
        }

        LOG_TRACE("> Reading zkey file");

        zkey = Zkey::PlonkZkeyHeader::loadPlonkZkeyHeader(fdZkey);

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
        // Precomputed 2 > evaluations buffer
        lengthPrecomputedBigBuffer += zkey->domainSize * 4 * 8;             // Evaluations QL, QR, QM, QO, QC, Sigma1, Sigma2, Sigma3
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

        evalPtr["Sigma1"] = polPtr["QC"] + zkey->domainSize;
        evalPtr["Sigma2"] = evalPtr["Sigma1"] + zkey->domainSize * 4;
        evalPtr["Sigma3"] = evalPtr["Sigma2"] + zkey->domainSize * 4;
        evalPtr["QL"] = evalPtr["Sigma3"] + zkey->domainSize * 4;
        evalPtr["QR"] = evalPtr["QL"] + zkey->domainSize * 4;
        evalPtr["QM"] = evalPtr["QR"] + zkey->domainSize * 4;
        evalPtr["QO"] = evalPtr["QM"] + zkey->domainSize * 4;
        evalPtr["QC"] = evalPtr["QO"] + zkey->domainSize * 4;
        evalPtr["lagrange"] = evalPtr["QC"] + zkey->domainSize * 4;

        PTau = (G1PointAffine *)(evalPtr["lagrange"] + zkey->domainSize * 4 * zkey->nPublic);

        // Read Q selectors polynomials and evaluations
        LOG_TRACE("... Loading QL, QR, QM, QO, & QC polynomial coefficients and evaluations");

        // Reserve memory for Q's polynomials
        polynomials["QL"] = new Polynomial<Engine>(E, polPtr["QL"], zkey->domainSize);
        polynomials["QR"] = new Polynomial<Engine>(E, polPtr["QR"], zkey->domainSize);
        polynomials["QM"] = new Polynomial<Engine>(E, polPtr["QM"], zkey->domainSize);
        polynomials["QO"] = new Polynomial<Engine>(E, polPtr["QO"], zkey->domainSize);
        polynomials["QC"] = new Polynomial<Engine>(E, polPtr["QC"], zkey->domainSize);

        int nThreads = omp_get_max_threads() / 2;

        // Read Q's polynomial coefficients from zkey file
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

        polynomials["QL"]->fixDegree();
        polynomials["QR"]->fixDegree();
        polynomials["QM"]->fixDegree();
        polynomials["QO"]->fixDegree();
        polynomials["QC"]->fixDegree();

        std::ostringstream ss;
        ss << "... Reading Q selector evaluations ";

        // Reserve memory for Q's evaluations
        evaluations["QL"] = new Evaluations<Engine>(E, evalPtr["QL"], zkey->domainSize * 4);
        evaluations["QR"] = new Evaluations<Engine>(E, evalPtr["QR"], zkey->domainSize * 4);
        evaluations["QM"] = new Evaluations<Engine>(E, evalPtr["QM"], zkey->domainSize * 4);
        evaluations["QO"] = new Evaluations<Engine>(E, evalPtr["QO"], zkey->domainSize * 4);
        evaluations["QC"] = new Evaluations<Engine>(E, evalPtr["QC"], zkey->domainSize * 4);

        // Read Q's evaluations from zkey file
        ThreadUtils::parcpy(evaluations["QL"]->eval,
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QL_SECTION) + zkey->domainSize,
                            sDomain * 4, nThreads);
        ThreadUtils::parcpy(evaluations["QR"]->eval,
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QR_SECTION) + zkey->domainSize,
                            sDomain * 4, nThreads);
        ThreadUtils::parcpy(evaluations["QM"]->eval,
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QM_SECTION) + zkey->domainSize,
                            sDomain * 4, nThreads);
        ThreadUtils::parcpy(evaluations["QO"]->eval,
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QO_SECTION) + zkey->domainSize,
                            sDomain * 4, nThreads);
        ThreadUtils::parcpy(evaluations["QC"]->eval,
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_QC_SECTION) + zkey->domainSize,
                            sDomain * 4, nThreads);

        // Read Sigma polynomial coefficients and evaluations from zkey file
        LOG_TRACE("... Loading Sigma1, Sigma2 & Sigma3 polynomial coefficients and evaluations");

        polynomials["Sigma1"] = new Polynomial<Engine>(E, polPtr["Sigma1"], zkey->domainSize);
        polynomials["Sigma2"] = new Polynomial<Engine>(E, polPtr["Sigma2"], zkey->domainSize);
        polynomials["Sigma3"] = new Polynomial<Engine>(E, polPtr["Sigma3"], zkey->domainSize);

        ThreadUtils::parcpy(polynomials["Sigma1"]->coef,
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_SIGMA_SECTION),
                            sDomain, nThreads);
        ThreadUtils::parcpy(polynomials["Sigma2"]->coef,
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_SIGMA_SECTION) + zkey->domainSize * 5,
                            sDomain, nThreads);
        ThreadUtils::parcpy(polynomials["Sigma3"]->coef,
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_SIGMA_SECTION) + zkey->domainSize * 10,
                            sDomain, nThreads);

        polynomials["Sigma1"]->fixDegree();
        polynomials["Sigma2"]->fixDegree();
        polynomials["Sigma3"]->fixDegree();

        evaluations["Sigma1"] = new Evaluations<Engine>(E, evalPtr["Sigma1"], zkey->domainSize * 4);
        evaluations["Sigma2"] = new Evaluations<Engine>(E, evalPtr["Sigma2"], zkey->domainSize * 4);
        evaluations["Sigma3"] = new Evaluations<Engine>(E, evalPtr["Sigma3"], zkey->domainSize * 4);

        ThreadUtils::parcpy(evaluations["Sigma1"]->eval,
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_SIGMA_SECTION) + zkey->domainSize,
                            sDomain * 4, nThreads);
        ThreadUtils::parcpy(evaluations["Sigma2"]->eval,
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_SIGMA_SECTION) + zkey->domainSize + zkey->domainSize * 5,
                            sDomain * 4, nThreads);
        ThreadUtils::parcpy(evaluations["Sigma3"]->eval,
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_SIGMA_SECTION) + zkey->domainSize + zkey->domainSize * 10,
                            sDomain * 4, nThreads);

        // Read Lagrange polynomials & evaluations from zkey file
        LOG_TRACE("... Loading Lagrange evaluations");
        evaluations["lagrange"] = new Evaluations<Engine>(E, evalPtr["lagrange"], zkey->domainSize * 4 * zkey->nPublic);
        for (uint64_t i = 0; i < zkey->nPublic; i++)
        {
            ThreadUtils::parcpy(evaluations["lagrange"]->eval + zkey->domainSize * 4 * i,
                                (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_LAGRANGE_SECTION) + zkey->domainSize + zkey->domainSize * 5 * i,
                                sDomain * 4, nThreads);
        }
        LOG_TRACE("... Loading Powers of Tau evaluations");

        // ThreadUtils::parset(PTau, 0, sizeof(G1PointAffine) * (zkey->domainSize + 6), nThreads);

        // domainSize + 6 corresponds to the SRS length stored in the zkey during the setup phase.
        // This represents the maximum SRS length required by the proving system.
        ThreadUtils::parcpy(this->PTau,
                            (G1PointAffine *)fdZkey->getSectionData(Zkey::ZKEY_PL_PTAU_SECTION),
                            (zkey->domainSize + 6) * sizeof(G1PointAffine), nThreads);

        // Load A, B & C map buffers
        LOG_TRACE("... Loading A, B & C map buffers");

        u_int64_t byteLength = sizeof(u_int32_t) * zkey->nConstraints;

        mapBuffersBigBuffer = new u_int32_t[zkey->nConstraints * 3];

        mapBuffers["A"] = mapBuffersBigBuffer;
        mapBuffers["B"] = mapBuffers["A"] + zkey->nConstraints;
        mapBuffers["C"] = mapBuffers["B"] + zkey->nConstraints;

        buffInternalWitness = new FrElement[zkey->nAdditions];

        LOG_TRACE("··· Loading additions");
        additionsBuff = (Zkey::Addition<Engine> *)fdZkey->getSectionData(Zkey::ZKEY_PL_ADDITIONS_SECTION);

        LOG_TRACE("··· Loading map buffers");
        // ThreadUtils::parset(mapBuffers["A"], 0, byteLength * 3, nThreads);

        // Read zkey sections and fill the buffers
        ThreadUtils::parcpy(mapBuffers["A"],
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_A_MAP_SECTION),
                            byteLength, nThreads);
        ThreadUtils::parcpy(mapBuffers["B"],
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_B_MAP_SECTION),
                            byteLength, nThreads);
        ThreadUtils::parcpy(mapBuffers["C"],
                            (FrElement *)fdZkey->getSectionData(Zkey::ZKEY_PL_C_MAP_SECTION),
                            byteLength, nThreads);

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
        // buffers.T    ← numArr, polynomials.R
        // buffers.Tz   ← denArr, polynomials.Wxi
        // polynomials.T ← buffers.Z, polynomials.Wxiw

        buffersLength = zkey->domainSize * 3; // Buffers A, B, C, tmp

        if (NULL == this->reservedMemoryPtr)
        {
            nonPrecomputedBigBuffer = new FrElement[lengthNonPrecomputedBigBuffer];
        }
        else
        {
            if (lengthNonPrecomputedBigBuffer * sizeof(FrElement) > reservedMemorySize)
            {
                ss.str("");
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
        buffers["numArr"] = buffers["T"];
        polPtr["R"] = buffers["T"];
        ;

        buffers["denArr"] = buffers["Tz"];
        polPtr["Wxi"] = buffers["Tz"];

        buffers["Z"] = polPtr["T"];
        polPtr["Wxiw"] = polPtr["T"];
        // }
        // catch (const std::exception &e)
        // {
        //     zklog.error("Plonk::setZkey() EXCEPTION: " + string(e.what()));
        //     exitProcess();
        // }
    }

    template <typename Engine>
    std::tuple<json, json> PlonkProver<Engine>::prove(BinFileUtils::BinFile *fdZkey, BinFileUtils::BinFile *fdWtns)
    {
        this->setZkey(fdZkey);
        return this->prove(fdWtns);
    }

    template <typename Engine>
    std::tuple<json, json> PlonkProver<Engine>::prove(BinFileUtils::BinFile *fdZkey, FrElement *buffWitness, WtnsUtils::Header *wtnsHeader)
    {
        this->setZkey(fdZkey);
        return this->prove(buffWitness, wtnsHeader);
    }

    template <typename Engine>
    std::tuple<json, json> PlonkProver<Engine>::prove(BinFileUtils::BinFile *fdWtns)
    {
        LOG_TRACE("> Reading witness file header");
        auto wtnsHeader = WtnsUtils::loadHeader(fdWtns);

        // Read witness data
        LOG_TRACE("> Reading witness file data");
        buffWitness = (FrElement *)fdWtns->getSectionData(2);

        return this->prove(buffWitness, wtnsHeader.get());
    }

    template <typename Engine>
    std::tuple<json, json> PlonkProver<Engine>::prove(FrElement *buffWitness, WtnsUtils::Header *wtnsHeader)
    {
        if (NULL == zkey)
        {
            throw std::runtime_error("Zkey data not set");
        }

        // try
        // {
        LOG_TRACE("PLONK PROVER STARTED");

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
        LOG_TRACE("  PLONK PROVE SETTINGS");
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
        json publicSignals;
        FrElement montgomery;
        for (u_int32_t i = 1; i <= zkey->nPublic; i++)
        {
            E.fr.toMontgomery(montgomery, buffWitness[i]);
            publicSignals.push_back(E.fr.toString(montgomery).c_str());
        }

        LOG_TRACE("PLONK PROVER FINISHED");

        ss.str("");
        ss << "Execution time: " << omp_get_wtime() - startTime << "\n";
        LOG_TRACE(ss);

        delete nonPrecomputedBigBuffer;

        return {proof->toJson(), publicSignals};
        // }
        // catch (const std::exception &e)
        // {
        //     zklog.error("Plonk::prove() EXCEPTION: " + string(e.what()));
        //     exitProcess();
        //     exit(-1);
        // }
    }

    template <typename Engine>
    void PlonkProver<Engine>::calculateAdditions()
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
    typename Engine::FrElement PlonkProver<Engine>::getWitness(u_int64_t idx)
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

    // ROUND 1
    template <typename Engine>
    void PlonkProver<Engine>::round1()
    {
        // STEP 1.1 - Generate random blinding scalars (b_1, ..., b9) ∈ F

        // 0 index not used, set to zero
        for (u_int32_t i = 1; i <= BLINDINGFACTORSLENGTH_PLONK; i++)
        {
            blindingFactors[i] = E.fr.one();
            // memset((void *)&(blindingFactors[i].v[0]), 0, sizeof(FrElement)); // TODO! Set to 1 to avoid zero
            // randombytes_buf((void *)&(blindingFactors[i].v[0]), sizeof(FrElement) - 1);
        }

        // STEP 1.2 - Compute wire polynomials a(X), b(X) and c(X)
        LOG_TRACE("> Computing A, B, C wire polynomials");
        computeWirePolynomials();

        // STEP 1.3 - Compute [a]_1, [b]_1, [c]_1
        G1Point A = multiExponentiation(polynomials["A"]);
        G1Point B = multiExponentiation(polynomials["B"]);
        G1Point C = multiExponentiation(polynomials["C"]);

        // First output of the prover is ([A]_1, [B]_1, [C]_1)
        proof->addPolynomialCommitment("A", A);
        proof->addPolynomialCommitment("B", B);
        proof->addPolynomialCommitment("C", C);
    }

    template <typename Engine>
    void PlonkProver<Engine>::computeWirePolynomials()
    {
        // Build A, B and C evaluations buffer from zkey and witness files
        FrElement bFactorsA[2] = {blindingFactors[2], blindingFactors[1]};
        FrElement bFactorsB[2] = {blindingFactors[4], blindingFactors[3]};
        FrElement bFactorsC[2] = {blindingFactors[5], blindingFactors[6]};

        computeWirePolynomial("A", bFactorsA);
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
    void PlonkProver<Engine>::computeWirePolynomial(std::string polName, FrElement blindingFactors[])
    {

// Compute all witness from signal ids and set them to the polynomial buffers
#pragma omp parallel for
        for (u_int32_t i = 0; i < zkey->nConstraints; ++i)
        {
            FrElement witness = getWitness(mapBuffers[polName][i]);
            E.fr.toMontgomery(buffers[polName][i], witness);
        }

        // Create the polynomial
        // and compute the coefficients of the wire polynomials from evaluations
        std::ostringstream ss;
        ss << "··· Computing " << polName << " ifft";
        LOG_TRACE(ss);
        polynomials[polName] = Polynomial<Engine>::fromEvaluations(E, fft, buffers[polName], polPtr[polName], zkey->domainSize, 2);

        // Compute the extended evaluations of the wire polynomials
        ss.str("");
        ss << "··· Computing " << polName << " fft";
        LOG_TRACE(ss);
        evaluations[polName] = new Evaluations<Engine>(E, fft, evalPtr[polName], *polynomials[polName], zkey->domainSize * 4);

        polynomials[polName]->blindCoefficients(blindingFactors, 2);
    }

    // ROUND 2
    template <typename Engine>
    void PlonkProver<Engine>::round2()
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
        // E.g1.copy(Commitment, *((G1PointAffine *)zkey->QC));
        // transcript->addPolCommitment(Commitment);
        E.g1.copy(Commitment, *((G1PointAffine *)zkey->S1));
        transcript->addPolCommitment(Commitment);
        E.g1.copy(Commitment, *((G1PointAffine *)zkey->S2));
        transcript->addPolCommitment(Commitment);
        E.g1.copy(Commitment, *((G1PointAffine *)zkey->S3));
        transcript->addPolCommitment(Commitment);

        // Add A to the transcript
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
        LOG_TRACE("> Computing Z multi exponentiation");
        G1Point Z = multiExponentiation(polynomials["Z"]);
        proof->addPolynomialCommitment("Z", Z);
    }

    template <typename Engine>
    void PlonkProver<Engine>::computeZ()
    {
        FrElement *numArr = buffers["numArr"];
        FrElement *denArr = buffers["denArr"];

        LOG_TRACE("··· Computing Z evaluations");

        std::ostringstream ss;
#pragma omp parallel for
        for (u_int64_t i = 0; i < zkey->domainSize; i++)
        {
            //            if ((0 != i) && (i % 100000 == 0)) {
            //                ss.str("");
            //                ss << "> Computing Z evaluation " << i << "/" << zkey->domainSize;
            //                //LOG_TRACE(ss);
            //            }

            FrElement omega = fft->root(zkeyPower, i);

            // Z(X) := numArr / denArr
            // numArr := (a + beta·ω + gamma)(b + beta·ω·k1 + gamma)(c + beta·ω·k2 + gamma)
            FrElement betaw = E.fr.mul(challenges["beta"], omega);

            FrElement num1 = buffers["A"][i];
            num1 = E.fr.add(num1, betaw);
            num1 = E.fr.add(num1, challenges["gamma"]);

            FrElement num2 = buffers["B"][i];
            num2 = E.fr.add(num2, E.fr.mul(*((FrElement *)zkey->k1), betaw));
            num2 = E.fr.add(num2, challenges["gamma"]);

            FrElement num3 = buffers["C"][i];
            num3 = E.fr.add(num3, E.fr.mul(*((FrElement *)zkey->k2), betaw));
            num3 = E.fr.add(num3, challenges["gamma"]);

            numArr[i] = E.fr.mul(num1, E.fr.mul(num2, num3));

            // denArr := (a + beta·sigma1 + gamma)(b + beta·sigma2 + gamma)(c + beta·sigma3 + gamma)
            FrElement den1 = buffers["A"][i];
            den1 = E.fr.add(den1, E.fr.mul(challenges["beta"], evaluations["Sigma1"]->eval[i * 4]));
            den1 = E.fr.add(den1, challenges["gamma"]);

            FrElement den2 = buffers["B"][i];
            den2 = E.fr.add(den2, E.fr.mul(challenges["beta"], evaluations["Sigma2"]->eval[i * 4]));
            den2 = E.fr.add(den2, challenges["gamma"]);

            FrElement den3 = buffers["C"][i];
            den3 = E.fr.add(den3, E.fr.mul(challenges["beta"], evaluations["Sigma3"]->eval[i * 4]));
            den3 = E.fr.add(den3, challenges["gamma"]);

            denArr[i] = E.fr.mul(den1, E.fr.mul(den2, den3));
        }

        FrElement numPrev = numArr[0];
        FrElement denPrev = denArr[0];
        FrElement numCur, denCur;

        for (u_int64_t i = 0; i < zkey->domainSize - 1; i++)
        {
            numCur = numArr[i + 1];
            denCur = denArr[i + 1];

            numArr[i + 1] = numPrev;
            denArr[i + 1] = denPrev;

            numPrev = E.fr.mul(numPrev, numCur);
            denPrev = E.fr.mul(denPrev, denCur);
        }

        numArr[0] = numPrev;
        denArr[0] = denPrev;

        // Compute the inverse of denArr to compute in the next command the
        // division numArr/denArr by multiplying num · 1/denArr
        batchInverse(denArr, zkey->domainSize);

// Multiply numArr · denArr where denArr was inverted in the previous command
#pragma omp parallel for
        for (u_int32_t i = 0; i < zkey->domainSize; i++)
        {
            buffers["Z"][i] = E.fr.mul(numArr[i], denArr[i]);
        }

        if (!E.fr.eq(buffers["Z"][0], E.fr.one()))
        {
            throw std::runtime_error("Copy constraints does not match");
        }

        // Compute polynomial coefficients z(X) from buffers.Z
        LOG_TRACE("··· Computing Z ifft");
        polynomials["Z"] = Polynomial<Engine>::fromEvaluations(E, fft, buffers["Z"], polPtr["Z"], zkey->domainSize, 3);

        // Compute extended evaluations of z(X) polynomial
        LOG_TRACE("··· Computing Z fft");
        evaluations["Z"] = new Evaluations<Engine>(E, fft, evalPtr["Z"], *polynomials["Z"], zkey->domainSize * 4);

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
    void PlonkProver<Engine>::round3()
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
        G1Point T1 = multiExponentiation(polynomials["T1"]);
        G1Point T2 = multiExponentiation(polynomials["T2"]);
        G1Point T3 = multiExponentiation(polynomials["T3"]);

        proof->addPolynomialCommitment("T1", T1);
        proof->addPolynomialCommitment("T2", T2);
        proof->addPolynomialCommitment("T3", T3);
    }

    template <typename Engine>
    void PlonkProver<Engine>::computeT()
    {
        LOG_TRACE("··· Computing T evaluations");

        std::ostringstream ss;
#pragma omp parallel for
        for (u_int64_t i = 0; i < zkey->domainSize * 4; i++)
        {
            FrElement omega = fft->root(zkeyPower + 2, i);
            FrElement omega2 = E.fr.square(omega);
            FrElement omegaW = E.fr.mul(omega, fft->root(zkeyPower, 1));
            FrElement omegaW2 = E.fr.square(omegaW);

            FrElement a = evaluations["A"]->eval[i];
            FrElement b = evaluations["B"]->eval[i];
            FrElement c = evaluations["C"]->eval[i];
            FrElement z = evaluations["Z"]->eval[i];
            FrElement zW = evaluations["Z"]->eval[(zkey->domainSize * 4 + 4 + i) % (zkey->domainSize * 4)];

            FrElement qm = evaluations["QM"]->eval[i];
            FrElement ql = evaluations["QL"]->eval[i];
            FrElement qr = evaluations["QR"]->eval[i];
            FrElement qo = evaluations["QO"]->eval[i];
            FrElement qc = evaluations["QC"]->eval[i];
            FrElement s1 = evaluations["Sigma1"]->eval[i];
            FrElement s2 = evaluations["Sigma2"]->eval[i];
            FrElement s3 = evaluations["Sigma3"]->eval[i];

            FrElement ap = E.fr.add(blindingFactors[2], E.fr.mul(blindingFactors[1], omega));
            FrElement bp = E.fr.add(blindingFactors[4], E.fr.mul(blindingFactors[3], omega));
            FrElement cp = E.fr.add(blindingFactors[6], E.fr.mul(blindingFactors[5], omega));

            FrElement zp = E.fr.add(E.fr.add(E.fr.mul(blindingFactors[7], omega2), E.fr.mul(blindingFactors[8], omega)),
                                    blindingFactors[9]);
            FrElement zWp = E.fr.add(
                E.fr.add(E.fr.mul(blindingFactors[7], omegaW2), E.fr.mul(blindingFactors[8], omegaW)),
                blindingFactors[9]);

            auto pi = E.fr.zero();
            for (u_int32_t j = 0; j < zkey->nPublic; j++)
            {
                const u_int32_t offset = (j * 4 * zkey->domainSize) + i;

                const auto lPol = evaluations["lagrange"]->getEvaluation(offset);
                const auto aVal = buffers["A"][j];
                pi = E.fr.sub(pi, E.fr.mul(lPol, aVal));
            }

            MulZ<Engine> mulz(E, fft);

            // T(X) := [ ( a(X)·b(X)·qm(X) + a(X)·ql(X) + b(X)·qr(X) + c(X)·qo(X) + PI(X) + qc(X) )
            //       +   ((a(X) + beta·X + gamma)(b(X) + beta·k1·X + gamma)(c(X) + beta·k2·X + gamma)·z(X)
            //       -    (a(X) + beta·sigma1(X) + gamma)(b(X) + beta·sigma2(X) + gamma)(c(X) + beta·sigma3(X) + gamma)z(Xω)) · alpha
            //       +    (z(X) - 1) · L_1(X) · alpha^2 ] · 1/Z_H(X)

            // e1 := a(X)b(X)qM(X) + a(X)qL(X) + b(X)qR(X) + c(X)qO(X) + PI(X) + qC(X)
            auto [e1, e1z] = mulz.mul2(a, b, ap, bp, i % 4);
            e1 = E.fr.mul(e1, qm);
            e1z = E.fr.mul(e1z, qm);

            e1 = E.fr.add(e1, E.fr.mul(a, ql));
            e1z = E.fr.add(e1z, E.fr.mul(ap, ql));

            e1 = E.fr.add(e1, E.fr.mul(b, qr));
            e1z = E.fr.add(e1z, E.fr.mul(bp, qr));

            e1 = E.fr.add(e1, E.fr.mul(c, qo));
            e1z = E.fr.add(e1z, E.fr.mul(cp, qo));

            e1 = E.fr.add(e1, pi);
            e1 = E.fr.add(e1, qc);

            // e2 := α[(a(X) + βX + γ)(b(X) + βk1X + γ)(c(X) + βk2X + γ)z(X)]
            auto betaw = E.fr.mul(challenges["beta"], omega);
            auto e2a = E.fr.add(a, betaw);
            e2a = E.fr.add(e2a, challenges["gamma"]);

            auto e2b = E.fr.add(b, E.fr.mul(betaw, *((FrElement *)zkey->k1)));
            e2b = E.fr.add(e2b, challenges["gamma"]);

            auto e2c = E.fr.add(c, E.fr.mul(betaw, *((FrElement *)zkey->k2)));
            e2c = E.fr.add(e2c, challenges["gamma"]);

            auto e2d = z;

            auto [e2, e2z] = mulz.mul4(e2a, e2b, e2c, e2d, ap, bp, cp, zp, i % 4);
            e2 = E.fr.mul(e2, challenges["alpha"]);
            e2z = E.fr.mul(e2z, challenges["alpha"]);

            // e3 := α[(a(X) + βSσ1(X) + γ)(b(X) + βSσ2(X) + γ)(c(X) + βSσ3(X) + γ)z(Xω)]
            auto e3a = a;
            e3a = E.fr.add(e3a, E.fr.mul(challenges["beta"], s1));
            e3a = E.fr.add(e3a, challenges["gamma"]);

            auto e3b = b;
            e3b = E.fr.add(e3b, E.fr.mul(challenges["beta"], s2));
            e3b = E.fr.add(e3b, challenges["gamma"]);

            auto e3c = c;
            e3c = E.fr.add(e3c, E.fr.mul(challenges["beta"], s3));
            e3c = E.fr.add(e3c, challenges["gamma"]);

            auto e3d = zW;
            auto [e3, e3z] = mulz.mul4(e3a, e3b, e3c, e3d, ap, bp, cp, zWp, i % 4);

            e3 = E.fr.mul(e3, challenges["alpha"]);
            e3z = E.fr.mul(e3z, challenges["alpha"]);

            // e4 := α^2(z(X)−1)L1(X)
            auto e4 = E.fr.sub(z, E.fr.one());
            e4 = E.fr.mul(e4, evaluations["lagrange"]->getEvaluation(i));
            e4 = E.fr.mul(e4, challenges["alpha2"]);

            auto e4z = E.fr.mul(zp, evaluations["lagrange"]->getEvaluation(i));
            e4z = E.fr.mul(e4z, challenges["alpha2"]);

            auto t = E.fr.add(E.fr.sub(E.fr.add(e1, e2), e3), e4);
            auto tz = E.fr.add(E.fr.sub(E.fr.add(e1z, e2z), e3z), e4z);

            buffers["T"][i] = t;
            buffers["Tz"][i] = tz;
        }

        // Compute the coefficients of the polynomial T2(X) from buffers.T2
        LOG_TRACE("··· Computing T ifft");
        polynomials["T"] = Polynomial<Engine>::fromEvaluations(E, fft, buffers["T"], polPtr["T"], zkey->domainSize * 4);

        // Divide the polynomial T2 by Z_H(X)
        polynomials["T"]->divZh(zkey->domainSize, 4);

        // Compute the coefficients of the polynomial T2z(X) from buffers.T2z
        LOG_TRACE("··· Computing Tz ifft");
        polynomials["Tz"] = Polynomial<Engine>::fromEvaluations(E, fft, buffers["Tz"], polPtr["Tz"], zkey->domainSize * 4);

        // Add the polynomial T2z to T2 to get the final polynomial T2
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
        polynomials["T1"]->fixDegree();

        polynomials["T2"] = new Polynomial<Engine>(E, polPtr["T2"], zkey->domainSize, 1);
        ThreadUtils::parcpy(polynomials["T2"]->coef, &polynomials["T"]->coef[zkey->domainSize], zkey->domainSize * sizeof(FrElement), nThreads);
        polynomials["T2"]->fixDegree();

        polynomials["T3"] = new Polynomial<Engine>(E, polPtr["T3"], zkey->domainSize, 6);
        ThreadUtils::parcpy(polynomials["T3"]->coef, &polynomials["T"]->coef[zkey->domainSize * 2], (zkey->domainSize + 6) * sizeof(FrElement), nThreads);
        polynomials["T3"]->fixDegree();

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
    void PlonkProver<Engine>::round4()
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
    void PlonkProver<Engine>::round5()
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

        LOG_TRACE("> Computing Wxi, Wxiw MSM");

        // The fifth output of the prover is ([Wxi]_1, [Wxiw]_1)
        G1Point commitWxi = multiExponentiation(polynomials["Wxi"]);
        G1Point commitWxiw = multiExponentiation(polynomials["Wxiw"]);
        Dump::Dump dump(E);
        dump.dump("commitWxi", commitWxi);
        dump.dump("commitWxiw", commitWxiw);

        proof->addPolynomialCommitment("Wxi", commitWxi);
        proof->addPolynomialCommitment("Wxiw", commitWxiw);
    }

    template <typename Engine>
    void PlonkProver<Engine>::computeR()
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
        for (u_int32_t i = 1; i <= zkey->nPublic; i++)
        {
            E.fr.toMontgomery(montgomery, buffWitness[i]);
            publicSignals[i] = montgomery;
        }

        for (u_int32_t i = 0; i < zkey->nPublic; i++)
        {
            const FrElement w = publicSignals[i+1];
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
    void PlonkProver<Engine>::computeWxi()
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
    void PlonkProver<Engine>::computeWxiw()
    {
        polynomials["Wxiw"] = polynomials["Z"];
        auto val = proof->getEvaluationCommitment("eval_zw");
        polynomials["Wxiw"]->subScalar(val);

        polynomials["Wxiw"]->divByZerofier(1, challenges["xiw"]);
    }

    template <typename Engine>
    void PlonkProver<Engine>::batchInverse(FrElement *elements, u_int64_t length)
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
    typename Engine::FrElement *PlonkProver<Engine>::polynomialFromMontgomery(Polynomial<Engine> *polynomial)
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
    typename Engine::G1Point PlonkProver<Engine>::multiExponentiation(Polynomial<Engine> *polynomial)
    {
        G1Point value;
        FrElement *pol = this->polynomialFromMontgomery(polynomial);

        E.g1.multiMulByScalar(value, PTau, (uint8_t *)pol, sizeof(pol[0]), polynomial->getDegree() + 1);

        return value;
    }
}
