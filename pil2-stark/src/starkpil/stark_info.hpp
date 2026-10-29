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

// everyRow, or the rows o + k*2^cycleBits for o in offsets (all other rows if complement).
struct Boundary
{
    std::string name;
    uint64_t cycleBits = 0;
    std::vector<uint64_t> offsets;
    bool complement = false;
};

// pil-info's MAX_DOMAIN_FACTORS: the GPU passes a domain's offsets by value.
constexpr uint64_t MAX_DOMAIN_OFFSETS = 32;

// log2 of the period of a validated boundary's zerofier on the extended domain.
inline uint64_t periodBits(const Boundary &b, uint64_t nBits, uint64_t nBitsExt)
{
    return nBitsExt - nBits + (b.name == "everyRow" ? 0 : b.cycleBits);
}

// shift^(N/2^c), the x^m of the zerofier's first extended row (c = 0 for everyRow), and the roots w(c)^o of V_O.
inline Goldilocks::Element domainRoots(const Boundary &b, uint64_t nBits, Goldilocks::Element *roots)
{
    if (b.name == "everyRow") return Goldilocks::exp(Goldilocks::shift(), uint64_t(1) << nBits);
    Goldilocks::Element shiftM = Goldilocks::shift();
    for (uint64_t i = 0; i < nBits - b.cycleBits; i++) Goldilocks::square(shiftM, shiftM);
    for (uint64_t k = 0; k < b.offsets.size(); k++) roots[k] = Goldilocks::exp(Goldilocks::w(b.cycleBits), b.offsets[k]);
    return shiftM;
}

// Empty if boundaries is a valid starkinfo list: everyRow first, then domains with sorted, distinct offsets in the cycle.
inline std::string validateBoundaries(const std::vector<Boundary> &boundaries, uint64_t nBits)
{
    if (boundaries.empty() || boundaries[0].name != "everyRow") return "boundary 0 must be everyRow";
    for (uint64_t i = 1; i < boundaries.size(); i++) {
        const Boundary &b = boundaries[i];
        std::string where = "boundary " + std::to_string(i) + ": ";
        if (b.name != "domain") return where + "unknown boundary " + b.name;
        if (b.cycleBits > nBits) return where + "cycle 2^" + std::to_string(b.cycleBits) + " exceeds N=2^" + std::to_string(nBits);
        if (b.offsets.empty()) return where + "a domain without offsets";
        if (b.offsets.size() > MAX_DOMAIN_OFFSETS) return where + "more than " + std::to_string(MAX_DOMAIN_OFFSETS) + " offsets";
        for (uint64_t k = 0; k < b.offsets.size(); k++) {
            if (b.offsets[k] >= (uint64_t(1) << b.cycleBits)) return where + "offset " + std::to_string(b.offsets[k]) + " is outside the cycle";
            // a repeated offset squares a factor of V_O, and Zh/V_O is no longer a polynomial
            if (k > 0 && b.offsets[k] <= b.offsets[k - 1]) return where + "offsets must be strictly increasing";
        }
    }
    return "";
}

inline bool inDomain(const Boundary &b, uint64_t row)
{
    if (b.name == "everyRow") return true;
    uint64_t r = row & ((uint64_t(1) << b.cycleBits) - 1);
    bool hit = false;
    for (uint64_t o : b.offsets) hit |= (o == r);
    return hit != b.complement;
}

// Places each boundary's zerofier table (one period) in the zi buffer; returns its size.
inline uint64_t ziLayout(const std::vector<Boundary> &boundaries, uint64_t nBits, uint64_t nBitsExt, std::vector<uint64_t> &offsets, std::vector<uint64_t> &masks)
{
    offsets.clear();
    masks.clear();
    uint64_t size = 0;
    for (const Boundary &b : boundaries) {
        uint64_t P = uint64_t(1) << periodBits(b, nBits, nBitsExt);
        offsets.push_back(size);
        masks.push_back(P - 1);
        size += P;
    }
    return size;
}

// FNV-1a of LE u64s: nBitsExt - nBits, then per boundary (isDomain, cycleBits, complement, |O|, O...). Matches exps-codegen.
inline uint64_t boundaryLayoutHash(const std::vector<Boundary> &boundaries, uint64_t nBits, uint64_t nBitsExt)
{
    uint64_t h = 0xcbf29ce484222325ull;
    auto word = [&h](uint64_t w) {
        for (int i = 0; i < 8; i++) {
            h ^= (w >> (8 * i)) & 0xff;
            h *= 0x100000001b3ull;
        }
    };
    word(nBitsExt - nBits);
    for (const Boundary &b : boundaries) {
        const bool domain = b.name != "everyRow";
        word(domain);
        word(domain ? b.cycleBits : 0);
        word(domain && b.complement);
        word(domain ? b.offsets.size() : 0);
        if (domain) for (uint64_t o : b.offsets) word(o);
    }
    return h;
}

/// The low-degree test run on the batched DEEP polynomial f_0.
enum class LowDegreeTestKind
{
    FRI,
    STIR, // Arnon–Chiesa–Fenzi–Yogev, ePrint 2024/390
};

/// STIR schedule, in the notation of the paper (Construction 5.2). Iteration i folds f_i by
/// 2^{k_i} and commits g_{i+1} on the next domain L_{i+1}, of half the size.
struct StirLayout;

class StirStruct
{
public:
    vector<uint64_t> foldingFactors; // k_i, in bits (length M)
    vector<uint64_t> logDegrees;     // log2 d_i, degree bound of f_i (length M+1)
    vector<uint64_t> logDomainSizes; // log2 |L_i| (length M+1)
    uint64_t numOodSamples = 1;      // s, out-of-domain samples per iteration — always 1, no longer on the wire
    vector<uint64_t> numQueries;     // t_i, shift queries into f_i (length M)
    vector<uint64_t> grindingBitsQueries; // grinding on iteration i's query message (length M)

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
    uint64_t grindingBitsQueries = 0; // FRI-only grinding (proof-of-work) budget on the query phase; 0 for STIR (see stir.grindingBitsQueries)

    // Which low-degree test the stark info selects. FRI's schedule is `logDomainSizes` (log2|L_i|
    // per round, STIR notation) + `numQueries`, STIR's is `stir`; the other one is left empty.
    LowDegreeTestKind lowDegreeTest = LowDegreeTestKind::FRI;
    uint64_t nQueries = 0;
    vector<uint64_t> logDomainSizes;
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

// Row-stripe count of the GPU evaluations (computeEvalsShifted grid.y). Shared with
// setMapOffsets, which sizes the "evals_partials" region for the reduction partials.
static constexpr uint64_t EVALS_HELPER_CHUNKS = 16;

class StarkInfo
{
public:
    // Read from starkInfo file
    StarkStruct starkStruct;

    uint64_t airgroupId;
    uint64_t airId;

    // Basic airs only: absorbed by the transcript after the stage-2 challenges.
    bool hasCircuitType = false;
    uint64_t circuitType = 0;

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
    // Boundary b's zerofier at extended row r is zi[ziOffsets[b] + (r & ziMasks[b])].
    vector<uint64_t> ziOffsets;
    vector<uint64_t> ziMasks;
    uint64_t ziSize = 0;

    uint64_t qDeg;
    uint64_t qDim;

    uint64_t nConstraints;

    uint64_t deepExpId;
    uint64_t cExpId;

    std::map<std::string, uint64_t> mapSectionsN;

    // Precomputed
    std::map<std::pair<std::string, bool>, uint64_t> mapOffsets;
    
    bool recursive = false;
    // GPU basic airs extend cm1/cm2 over their small traces (recursive airs need the cm1 root first).
    bool inplaceStageCommit = false;
    // GPU GL commit trees (cm*, const) are stored from level 1; queries rehash level 0.
    bool dropLeafLevel = false;
    bool verify_constraints = false;
    bool verify = false;
    bool gpu = false;

    bool calculateFixedExtended = false;

    uint64_t mapTotalN;
    uint64_t mapTotalNCustomCommitsFixed;
    
    uint64_t maxNBlocks, nrowsPack;
    
    uint64_t proofSize;

    uint64_t airValuesSize;
    uint64_t airgroupValuesSize;
    uint64_t proofValuesSize;

    /* Constructor */
    StarkInfo(string file, bool recursive_final = false, bool recursive = false, bool verify_constraints = false, bool verify = false, bool gpu = false);
    StarkInfo() {};

    uint64_t getTraceLandingOffset();
    uint64_t getNumNodesMTCommit(uint64_t height);
    /* Offset in a commit tree's stored nodes of the level last_levels copies, and its node count */
    uint64_t getLastLevelOffset(uint64_t height, uint64_t &nNodes);
    
    /* Loads data from a json object */
    void load (json j);

    bool hasConstants() const { return nConstants > 0; }
    /// Trees opened at the first query layer: stages 1..nStages+1, the const tree if any, custom commits.
    uint64_t numInitialTrees() const { return nStages + 1 + (hasConstants() ? 1 : 0) + customCommits.size(); }

    /// Number of Merkle trees the low-degree test commits, and the leaf width of each.
    uint64_t numLowDegreeTestTrees() const;
    uint64_t lowDegreeTestTreeWidth(uint64_t i) const;

    /// u64 the flat proof buffer spends on the STIR section.
    uint64_t stirProofSectionSize() const;

    void setMapOffsets();
    StirLayout stirLayout() const; // valid after setMapOffsets on a GPU STIR stark info
    uint64_t stirBase = 0;         // where setMapOffsets placed the STIR regions

    void setMemoryExpressions(uint64_t nTmp1, uint64_t nTmp3);

    void getProofSize();

    uint64_t getPinnedProofSize();

    uint64_t getNumNodesMT(uint64_t height);
};

#endif