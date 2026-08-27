#ifndef STARK_INFO_HPP
#define STARK_INFO_HPP

#include <nlohmann/json.hpp>
#include <cmath>
#include <string>
#include <vector>
#include "zkassert.hpp"
#include "goldilocks_base_field.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "eval_info.hpp"

using json = nlohmann::json;
using namespace std;

/* StarkInfo class contains the contents of the file zkevm.starkinfo.json,
   which is parsed during the constructor */

typedef enum
{
    const_ = 0,
    cm = 1,
    tmp = 2,
    public_ = 3,
    airgroupvalue = 4,
    challenge = 5,
    number = 6,
    string_ = 7,
    airvalue = 8,
    proofvalue = 9,
    custom = 10,
    x = 11,
    Zi = 12,
    eval = 13,
    xDivXSubXi = 14,
    q = 15,
    f = 16,
} opType;


class CustomCommits
{
public:
    std::string name;
    vector<uint32_t> stageWidths;
    vector<uint32_t> publicValues;
};

class Boundary
{
public:
    std::string name;
    uint64_t offsetMin;
    uint64_t offsetMax;
};

class StepStruct
{
public:
    uint64_t nBits;
};

/// The low-degree test run on the batched DEEP polynomial f_0.
enum class LowDegreeTestKind
{
    FRI,
    STIR, // Arnon–Chiesa–Fenzi–Yogev, ePrint 2024/390
};

/// STIR schedule, in the notation of the paper (Construction 5.2). Iteration i folds f_i by
/// 2^{k_i} and commits g_{i+1} on the next domain L_{i+1}, of half the size.
class StirStruct
{
public:
    vector<uint64_t> foldingFactors; // k_i, in bits (length M)
    vector<uint64_t> logDegrees;     // log2 d_i, degree bound of f_i (length M+1)
    vector<uint64_t> logDomainSizes; // log2 |L_i| (length M+1)
    uint64_t numOodSamples;          // s, out-of-domain samples per iteration
    vector<uint64_t> numQueries;     // t_i, shift queries into f_i (length M)
    vector<uint64_t> grindingBits;   // grinding on iteration i's query message (length M)

    uint64_t numIterations() const { return foldingFactors.size(); }
};

class StarkStruct
{
public:
    uint64_t nBits;
    uint64_t nBitsExt;
    bool hashCommits;
    string verificationHashType;
    uint64_t lastLevelVerification;
    uint64_t merkleTreeArity;
    bool merkleTreeCustom;
    uint64_t transcriptArity;
    uint64_t powBits;

    // Which low-degree test the stark info selects. FRI's schedule is `steps`/`nQueries`, STIR's
    // is `stir`; the other one is left empty.
    LowDegreeTestKind lowDegreeTest = LowDegreeTestKind::FRI;
    uint64_t nQueries = 0;
    vector<StepStruct> steps;
    StirStruct stir;
};

/// Merkle path levels one query publishes: the tree's depth less the levels the published last
/// level replaces. Saturating, because `lastLevelVerification` can exceed a short tree's height
/// and every caller works in unsigned arithmetic.
inline uint64_t merkleProofLevels(uint64_t nBits, uint64_t arity, uint64_t lastLevelVerification, bool bn128)
{
    if (nBits == 0) return 0;
    uint64_t levels = bn128 ? (uint64_t)std::floor((nBits - 1) / std::ceil(std::log2(arity))) + 1
                            : (uint64_t)std::ceil(nBits / std::log2(arity));
    return levels > lastLevelVerification ? levels - lastLevelVerification : 0;
}

opType string2opType (const string s);
string opType2string (const opType op);

class PolMap
{
public:
    uint64_t stage;
    std::string name;
    vector<uint64_t> lengths;
    uint64_t dim;
    bool imPol;
    uint64_t stagePos;
    uint64_t stageId;
    uint64_t commitId;
    uint64_t expId;
    uint64_t polsMapId;
};

class EvMap
{
public:
    typedef enum
    {
        cm = 0,
        _const = 1,
        custom = 2,
    } eType;

    eType type;
    uint64_t id;
    int64_t prime;
    uint64_t commitId;
    uint64_t openingPos;

    void setType (string s)
    {
        if (s == "cm") type = cm;
        else if (s == "const") type = _const;
        else if (s == "custom") type = custom;
        else
        {
            zklog.error("EvMap::setType() found invalid type: " + s);
            exitProcess();
        }
    }
};

class StarkInfo
{
public:
    // Read from starkInfo file
    StarkStruct starkStruct;

    uint64_t airgroupId;
    uint64_t airId;

    uint64_t nPublics;
    uint64_t nConstants;
    
    uint64_t nStages;
    uint64_t maxProofBuffSize;
    uint64_t maxProofSize;
    uint64_t maxTreeWidth;
    
    vector<CustomCommits> customCommits;

    vector<PolMap> cmPolsMap;
    vector<PolMap> constPolsMap;
    vector<PolMap> challengesMap;
    vector<PolMap> airgroupValuesMap;
    vector<PolMap> airValuesMap;
    vector<PolMap> proofValuesMap;
    vector<PolMap> publicsMap;
    vector<vector<PolMap>> customCommitsMap;

    vector<EvMap> evMap;
    
    vector<int64_t> openingPoints;
    vector<Boundary> boundaries;

    uint64_t qDeg;
    uint64_t qDim;

    uint64_t nConstraints;

    uint64_t friExpId;
    uint64_t cExpId;

    std::map<std::string, uint64_t> mapSectionsN;

    // Precomputed
    std::map<std::pair<std::string, bool>, uint64_t> mapOffsets;
    
    bool recursive = false;
    bool verify_constraints = false;
    bool verify = false;
    bool gpu = false;
    bool preallocate = false;
    // Table air: proved at most once, so nothing reuses its unpacked const pols.
    bool singleUse = false;

    bool calculateFixedExtended = false;
    // ("const", false) aliases the node area of ("const", true) instead of getting its own
    // region. Implies calculateFixedExtended, which makes extendAndMerkelizeFixed the last
    // reader of the const pols before the nodes overwrite them. Forbids const reuse across
    // proofs: afterwards the region holds Merkle nodes.
    bool constPolsAliasTree = false;

    uint64_t mapTotalN;
    uint64_t mapTotalNContributions;
    uint64_t mapTotalNCustomCommitsFixed;
    
    uint64_t maxNBlocks, nrowsPack;
    
    uint64_t proofSize;

    uint64_t airValuesSize;
    uint64_t airgroupValuesSize;
    uint64_t proofValuesSize;

    /* Constructor */
    StarkInfo(string file, bool recursive_final = false, bool recursive = false, bool verify_constraints = false, bool verify = false, bool gpu = false, bool preallocate = false, bool single_use = false);
    StarkInfo() {};
    
    /* Loads data from a json object */
    void load (json j);

    /// The prover, verifier and proof layout only implement FRI so far: stop with a clear message
    /// rather than misread a STIR stark info as a FRI one.
    void requireFri(const std::string &what) const;

    void setMapOffsets();

    void setMemoryExpressions(uint64_t nTmp1, uint64_t nTmp3);

    void getProofSize();

    uint64_t getPinnedProofSize();

    uint64_t getNumNodesMT(uint64_t height);
};

#endif