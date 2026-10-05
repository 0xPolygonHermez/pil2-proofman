#ifndef PILFFLONK_INFO_HPP
#define PILFFLONK_INFO_HPP

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace PilFflonk {

// <air>.pilfflonkinfo.json (pilfflonk/docs/formats.md#pilfflonkinfo): what the prover knows of an
// AIR, the part of the STARK's starkinfo.json pilfflonk keeps and the layout of the AIR's f_i. The
// format is owned by the Rust crate proofman-pilfflonk (pilfflonk/src/pilfflonk_info.rs), which
// writes the file for the setup and documents every field; the fields here keep its names and
// meaning. BN128 has no extension field: every dim is 1.
//
// The reader checks the types of all the fields and that no field is missing or unknown, so that
// the two sides cannot drift apart silently (pilfflonk/docs/README.md#code-map), and every index
// the prover will follow: into the pol maps, openingPoints and the stages. The rest of the
// format's rules (those of the layout and of the evMap against it,
// pilfflonk/docs/protocol.md#layout) are the Rust type's to check when it writes the file.

struct PolMapEntry {
    uint64_t stage;
    std::string name;
    uint64_t dim;
    uint64_t polsMapId;
    uint64_t stageId;
    std::vector<uint64_t> lengths; // empty unless an element of an array column
    bool imPol;
    bool hasExpId;
    uint64_t expId; // if hasExpId
    uint64_t stagePos;
};

struct ChallengeMapEntry {
    std::string name;
    uint64_t stage;
    uint64_t dim;
    uint64_t stageId;
};

// An entry of airValuesMap or airgroupValuesMap.
struct NameStageEntry {
    std::string name;
    uint64_t stage;
    std::vector<uint64_t> lengths;
};

enum class PolType {
    Cm,    // cmPolsMap, "cm"
    Const, // constPolsMap, "const"
};

// Column id of type at ξ·ω^prime.
struct EvMapEntry {
    PolType type;
    uint64_t id;
    int64_t prime;
    uint64_t openingPos; // openingPoints[openingPos] == prime
};

enum class BoundaryType { EveryRow, FirstRow, LastRow, EveryFrame };

struct Boundary {
    BoundaryType type;
    uint64_t offsetMin; // EveryFrame only, 0 otherwise
    uint64_t offsetMax; // EveryFrame only, 0 otherwise
};

struct LayoutPol {
    uint64_t id; // in constPolsMap for an f of stage 0, in cmPolsMap otherwise
    std::string name;
};

// f(X) = Σ_{j<k} pols[j](X^k)·X^j, opened at the roots of ξ·ω^s for each s of offsets.
struct LayoutEntry {
    uint64_t stage; // 0 fixed, 1 … nStages committed, nStages + 1 Q
    std::vector<LayoutPol> pols;
    uint64_t k; // pols.size()
    std::vector<int64_t> offsets;
    uint64_t degree; // bound on the number of coefficients of f
};

class PilfflonkInfo {
public:
    // Reads the file at path. Throws IoError if it cannot be read, and FormatError, naming the
    // file and the field, if it is not valid JSON or not a pilfflonkinfo.
    static PilfflonkInfo load(const std::string &path);

    // The same, from the file's text.
    static PilfflonkInfo parse(const std::string &text);

    uint64_t qStage() const { return nStages + 1; }

    std::string name;
    uint64_t airgroupId;
    uint64_t airId;
    uint64_t nBits;
    uint64_t nStages;
    uint64_t nConstants;
    std::vector<PolMapEntry> cmPolsMap;
    std::vector<PolMapEntry> constPolsMap;
    std::vector<ChallengeMapEntry> challengesMap;
    std::vector<NameStageEntry> airValuesMap;
    std::vector<NameStageEntry> airgroupValuesMap;
    std::map<std::string, uint64_t> mapSectionsN;
    std::vector<int64_t> openingPoints;
    std::vector<Boundary> boundaries;
    std::vector<EvMapEntry> evMap;
    uint64_t qDeg;
    uint64_t qDim; // 1
    uint64_t maxQDegree;
    uint64_t cExpId;
    std::vector<LayoutEntry> layout;
};

} // namespace PilFflonk

#endif
