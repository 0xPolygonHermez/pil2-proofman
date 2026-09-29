#ifndef PILFFLONK_PROVING_KEY_HPP
#define PILFFLONK_PROVING_KEY_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "alt_bn128.hpp"
#include "pilfflonk_commit.hpp"
#include "pilfflonk_expressions.hpp"
#include "pilfflonk_expressions_bin.hpp"
#include "pilfflonk_info.hpp"
#include "pilfflonk_lde.hpp"
#include "pilfflonk_srs.hpp"

namespace PilFflonk {

// What the prover reads of pilout.globalInfo.json (spec A.6): where the AIRs' files are, and the
// counts of the values every instance is given. The file's owner is the Rust type
// proofman_pilfflonk::PilfflonkGlobalInfo (pilfflonk/src/global_info.rs), which validates all of it
// before the prover runs; this reader checks the fields it uses (their types, and that the file is
// a pilfflonk one of format version 1) and ignores the rest.
struct GlobalInfo {
    struct Air {
        std::string name;
        uint64_t numRows;
    };

    std::string name;
    std::vector<std::string> airGroups;
    std::vector<std::vector<Air>> airs; // by airgroup, in the pilout's order
    uint64_t nPublics = 0;
    std::vector<uint64_t> proofValueStages; // the stage of each entry of proofValuesMap

    // Throws IoError if the file cannot be read, FormatError if it is not such a file.
    static GlobalInfo load(const std::string &path);
    static GlobalInfo parse(const std::string &text);
};

// Where a column is committed: layout entry `f` of its AIR, as its polynomial p_j of index j.
struct LayoutPosition {
    uint64_t f;
    uint64_t j;
};

// The degrees of A.1 for an AIR, derived from its pilfflonkinfo as the setup derives them
// (setup/pilfflonk/src/layout.rs; pilfflonk/src/degrees.rs in Rust): they are not stored.
struct AirDegrees {
    uint64_t n;           // N = 2^nBits
    uint64_t maxOpenings; // |O|_max: the most offsets of an f of a committed stage (1 … nStages)
    uint64_t qCoefficients; // Q's bound, qDeg·N + (qDeg+1)·|O|_max + 1
    uint64_t nBitsExt;      // the smallest power of two >= Q's bound and >= N + |O|_max + 1
};

// Throws FormatError, naming `name`, if nBitsExt exceeds 28.
AirDegrees airDegrees(const PilfflonkInfo &info, const std::string &name);

// A column an expression's code reads: an operand (type, arg1) with bin.types().isColumn(type),
// so type 0 is a fixed column (arg1 its constPolsMap index) and type s >= 1 the committed column of
// stage s at stagePos arg1.
struct ColumnRead {
    uint32_t type;
    uint32_t index;
    bool operator<(const ColumnRead &o) const { return type != o.type ? type < o.type : index < o.index; }
    bool operator==(const ColumnRead &o) const { return type == o.type && index == o.index; }
};

// The columns expression expId of `bin` reads, each once, in increasing order. Throws
// std::invalid_argument if bin has no such expression.
std::vector<ColumnRead> columnsRead(const ExpressionsBin &bin, uint64_t expId);

// The proving key of one AIR (spec §4.4, step 1): its pilfflonkinfo, its bytecode, and its fixed
// columns, both on H (for the intermediate polynomials) and as polynomials (for Q's coset and the
// opening), with what the prover derives from them. Immutable once built.
//
// Checks, besides what each reader checks: the files agree (the .bin's stages and the .const's size
// are the pilfflonkinfo's, and the rows of the .bin's constraints lie in the trace), and what this
// prover supports: Q not split, and a layout that packs every committed column once, Q in an f of
// its own with k = 1.
class AirKey {
public:
    // From the files' contents. `name` is what the errors call the AIR. Throws FormatError.
    AirKey(PilfflonkInfo info, ExpressionsBin bin, const uint8_t *constants, uint64_t constantsBytes,
           const std::string &name);

    // Reads <dir>/<name>.pilfflonkinfo.json, <dir>/<name>.bin and <dir>/<name>.const. Throws
    // IoError and FormatError.
    static std::unique_ptr<AirKey> load(const std::string &dir, const std::string &name);

    AirKey(const AirKey &) = delete;
    AirKey &operator=(const AirKey &) = delete;

    const std::string &name() const { return airName; }
    const PilfflonkInfo &info() const { return pilfflonkInfo; }
    const ExpressionsBin &bin() const { return expressionsBin; }
    const Expressions &expressions() const { return *interpreter; }
    const AirDegrees &degrees() const { return airDegrees_; }
    // N and N' = 2^nBitsExt.
    const Lde &lde() const { return *extension; }
    uint64_t n() const { return airDegrees_.n; }

    // Fixed column c (constPolsMap index) on H, N values in natural order.
    const FrElement *fixedEvaluations(uint64_t c) const { return fixedEvals.get() + c * n(); }
    // Its interpolant, of N coefficients. Not const, as rapidsnark's API takes it, but never changed.
    Poly *fixedPolynomial(uint64_t c) const { return fixedPolys[c].get(); }

    // Where the layout commits constPolsMap[id] or cmPolsMap[id]; f = UINT64_MAX if it does not
    // (a column the evMap never opens, spec A.2).
    const LayoutPosition &constPosition(uint64_t id) const { return constPositions[id]; }
    const LayoutPosition &cmPosition(uint64_t id) const { return cmPositions[id]; }
    static constexpr uint64_t NOT_COMMITTED = UINT64_MAX;

    // The coefficients of b(X) of the blinding of every column of f (spec A.3): |O_f| + 1 for an f
    // of a committed stage, 0 for a fixed one and for Q's (not split).
    uint64_t blindLength(uint64_t f) const;

    // The witness of an instance: the stagePos in stage 1 of each of its C columns, column c being
    // the cmPolsMap entry of stage 1 that is not an im pol and has stageId c (spec A.6).
    const std::vector<uint64_t> &witnessColumns() const { return witness; }

    // The cmPolsMap index of the column of stage s (1 … nStages + 1) at stagePos p: cmIds()[s][p].
    const std::vector<std::vector<uint64_t>> &cmIds() const { return cmIdsByStage; }

    // The columns the code of Q (cExpId) reads: fixed ones and committed ones of stages 1 … nStages,
    // each committed by the layout (checked when the key is built).
    const std::vector<ColumnRead> &qReads() const { return qColumns; }

    // The number of f of the layout of stage 0 (the fixed ones), which come first (spec A.5).
    uint64_t nFixedF() const { return nFixed; }

    // The layout entry of Q (not split).
    uint64_t qF() const { return qEntry; }

private:
    std::string airName;
    PilfflonkInfo pilfflonkInfo;
    ExpressionsBin expressionsBin;
    std::unique_ptr<Expressions> interpreter; // refers to expressionsBin
    AirDegrees airDegrees_;
    std::unique_ptr<Lde> extension;
    std::unique_ptr<FrElement[]> fixedEvals;
    std::unique_ptr<FrElement[]> fixedCoefs;
    std::vector<std::unique_ptr<Poly>> fixedPolys;
    std::vector<LayoutPosition> constPositions;
    std::vector<LayoutPosition> cmPositions;
    std::vector<uint64_t> witness;
    std::vector<std::vector<uint64_t>> cmIdsByStage;
    std::vector<ColumnRead> qColumns;
    uint64_t nFixed = 0;
    uint64_t qEntry = 0;
};

// The proving key of a proof (spec §4.4, step 1; §4.2.6's provingKey/): the globalInfo, the SRS and
// every AIR's key. Immutable once built: proofs may share it, from several threads.
class ProvingKey {
public:
    ProvingKey(GlobalInfo globalInfo, Srs srs, std::vector<std::vector<std::unique_ptr<AirKey>>> airs);

    // Reads the provingKey/ at dir, as setup-pilfflonk writes it (spec §4.2.6):
    //   <dir>/pilout.globalInfo.json
    //   <dir>/<name>/pilfflonk/pilfflonk.srs.bin
    //   <dir>/<name>/<airgroup>/airs/<air>/air/<air>.{pilfflonkinfo.json, bin, const}
    // The vkey is not read: the digest the transcript absorbs is the orchestrator's (spec A.4), which
    // reads and checks the vkey. Throws IoError and FormatError, and FormatError if an AIR's layout
    // needs more powers [τ^i]₁ than the SRS holds or its pilfflonkinfo is not the globalInfo's AIR.
    static std::unique_ptr<ProvingKey> load(const std::string &dir);

    ProvingKey(const ProvingKey &) = delete;
    ProvingKey &operator=(const ProvingKey &) = delete;

    const GlobalInfo &globalInfo() const { return info; }
    const Srs &srs() const { return structuredReferenceString; }

    // The key of air airId of airgroup airgroupId. Throws std::invalid_argument if there is none.
    const AirKey &air(uint64_t airgroupId, uint64_t airId) const;

private:
    GlobalInfo info;
    Srs structuredReferenceString;
    std::vector<std::vector<std::unique_ptr<AirKey>>> airKeys;
};

} // namespace PilFflonk

#endif
