#ifndef PLONK_PROVER_HPP
#define PLONK_PROVER_HPP

#include <string>
#include <map>
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

#define BLINDINGFACTORSLENGTH_PLONK 11

namespace Plonk {

    template<typename Engine>
    class PlonkProver {
        using FrElement = typename Engine::FrElement;
        using G1Point = typename Engine::G1Point;
        using G1PointAffine = typename Engine::G1PointAffine;

        Engine &E;
        FFT<typename Engine::Fr> *fft = NULL;

        Zkey::PlonkZkeyHeader *zkey;
        u_int32_t zkeyPower;
        std::string curveName;
        size_t sDomain;

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

        FrElement *inverses;
        FrElement *products;

        // This is the length of the buffer that must be zeroed after each proof (starting from buffers["A"] pointer)
        u_int64_t buffersLength;

        std::map<std::string, FrElement *> polPtr;
        std::map<std::string, FrElement *> evalPtr;

        std::map<std::string, u_int32_t *> mapBuffers;
        std::map<std::string, FrElement *> buffers;
        std::map<std::string, Polynomial<Engine> *> polynomials;
        std::map<std::string, Evaluations<Engine> *> evaluations;

        std::map <std::string, FrElement> toInverse;
        std::map <std::string, FrElement> challenges;

        FrElement blindingFactors[BLINDINGFACTORSLENGTH_PLONK + 1];

        Keccak256Transcript<Engine> *transcript;
        SnarkProof<Engine> *proof;
    public:
        PlonkProver(Engine &E);
        PlonkProver(Engine &E, void* reservedMemoryPtr, uint64_t reservedMemorySize);

        ~PlonkProver();

        void setZkey(BinFileUtils::BinFile *fdZkey);

        std::tuple <json, json, std::vector<uint8_t>> prove(BinFileUtils::BinFile *fdZkey, BinFileUtils::BinFile *fdWtns);
        std::tuple <json, json, std::vector<uint8_t>> prove(BinFileUtils::BinFile *fdZkey, FrElement *wtns, WtnsUtils::Header* wtnsHeader = NULL);

        std::tuple <json, json, std::vector<uint8_t>> prove(BinFileUtils::BinFile *fdWtns);
        std::tuple <json, json, std::vector<uint8_t>> prove(FrElement *wtns, WtnsUtils::Header* wtnsHeader = NULL);

    protected:
        void initialize(void* reservedMemoryPtr, uint64_t reservedMemorySize = 0);

        void removePrecomputedData();

        void calculateAdditions();

        FrElement getWitness(u_int64_t idx);

        void round1();

        void round2();

        void round3();

        void round4();

        void round5();

        void computeWirePolynomials();

        void computeWirePolynomial(std::string polName, FrElement blindingFactors[]);

        void computeZ();

        void computeT();

        void computeR();

        void computeWxi();

        void computeWxiw();

        void batchInverse(FrElement *elements, u_int64_t length);

        FrElement *polynomialFromMontgomery(Polynomial<Engine> *polynomial);

        G1Point multiExponentiation(Polynomial<Engine> *polynomial);
    };
}

#include "plonk_prover.c.hpp"

#endif
