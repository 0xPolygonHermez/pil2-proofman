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
#include "pilfflonk_gpu.hpp"
#include "pilfflonk_info.hpp"
#include "pilfflonk_lde.hpp"
#include "pilfflonk_srs.hpp"

namespace PilFflonk {

class GpuKey;    // pilfflonk_key_gpu.hpp
class GpuAirKey; // pilfflonk_key_gpu.hpp

// What the prover reads of pilout.globalInfo.json (pilfflonk/docs/formats.md#globalinfo): where the
// AIRs' files are, and the counts of the values every instance is given. The file's owner is the
// Rust type proofman_pilfflonk::PilfflonkGlobalInfo (pilfflonk/src/global_info.rs), which validates
// all of it before the prover runs; this reader checks the fields it uses (their types, and that
// the file is a pilfflonk one of format version 1) and ignores the rest.
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

// The degrees of an AIR (pilfflonk/docs/protocol.md#degrees), derived from its pilfflonkinfo as the
// setup derives them (setup/pilfflonk/src/layout.rs; pilfflonk/src/degrees.rs in Rust, Degrees and
// QSplit): they are not stored.
struct AirDegrees {
    uint64_t n;           // N = 2^nBits
    uint64_t maxOpenings; // |O|_max: the most offsets of an f of a committed stage (1 … nStages)
    uint64_t qCoefficients; // Q's bound, qDeg·N + (qDeg+1)·|O|_max + 1
    uint64_t nBitsExt;      // the smallest power of two >= Q's bound and >= N + |O|_max + 1
    // The pieces Q_0 … Q_{m−1} Q is committed as (pilfflonk/docs/protocol.md#q-pieces),
    // Q(X) = Σ_i X^(i·qStride)·Q_i(X): m = ⌈qDeg/maxQDegree⌉ if 0 < maxQDegree < qDeg, and otherwise
    // 1, Q itself, with qStride 0. Split, qStride = maxQDegree·N, and piece i holds the coefficients
    // i·qStride … (i+1)·qStride − 1 of Q (the last one those up to Q's bound), and each boundary two
    // random ones that cancel: b0·X^qStride + b1·X^(qStride+1) added to the piece below it,
    // b0 + b1·X subtracted from the one above.
    uint64_t qStride;
    // The bound on the coefficients of each piece: qStride + 2 but the last, qCoefficients −
    // (m−1)·qStride.
    std::vector<uint64_t> qPieceCoefficients;
};

// Throws FormatError, naming `name`, if nBitsExt exceeds 28, or maxQDegree is not 0 while it does not
// split Q (maxQDegree >= qDeg: the setup writes 0 then).
AirDegrees airDegrees(const PilfflonkInfo &info, const std::string &name);

// The coefficients of Q that piece i holds before its boundaries are blinded (AirDegrees): `length`
// of them from `start` = i·qStride, qStride but in the last piece, which holds those up to Q's bound.
struct QPieceRange {
    uint64_t start;
    uint64_t length;
};

QPieceRange qPieceRange(const AirDegrees &d, uint64_t i);

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

// An operand of a std prover hint the prover reads on H: one of those the STARK's addHintField takes
// (hints.cpp), but for an air value (v1 has none, pilfflonk/docs/README.md#scope).
struct HintInput {
    enum class Kind { Column, Expression, Number };
    Kind kind = Kind::Number;
    ColumnRead column{0, 0}; // Column: a fixed one (type 0) or a committed one of an earlier stage
    int64_t offset = 0;      // Column: read at row i + offset, cyclically
    uint64_t expId = 0;      // Expression: its code in the .bin's section 1
    FrElement number;        // Number, in Montgomery form
};

// A prover hint of the std in the .bin (section 3, pilfflonk/docs/formats.md#bytecode): the column
// it gives, reference, of stage `stage` >= 2 at stagePos (cmPolsMap[cmId]), from the quotient
// numerator/denominator on each row of H (pilfflonk/docs/protocol.md#hint-columns):
// - im_col (Kind::ImCol), with the fields numerator and denominator: the column is the quotient
//   itself, as the STARK's calculateImHints computes it with multiplyHintFields;
// - gprod_col and gsum_col (Kind::Prod, Kind::Sum), with numerator_air and denominator_air: the
//   column accumulates the quotient, a running product or a running sum, as the STARK's
//   calculateWitnessSTD computes it with accMulHintFields.
struct StdHint {
    // In the order the prover computes the hints of a stage, the STARK's (gen_proof.hpp): the im_col
    // hints (calculateImHints), then the gprod_col ones and the gsum_col ones (calculateWitnessSTD).
    enum class Kind { ImCol, Prod, Sum };

    uint64_t hint; // its index in the .bin's hints
    std::string name;
    Kind kind;
    uint64_t stage;
    uint64_t stagePos;
    uint64_t cmId;
    HintInput numerator;
    HintInput denominator;
};

// The proving key of one AIR (pilfflonk/docs/protocol.md#proof-sequence, step 1): its
// pilfflonkinfo, its bytecode, and its fixed columns, both on H (for the intermediate polynomials)
// and as polynomials (for Q's coset and the opening), with what the prover derives from them.
// Immutable once built.
//
// Checks, besides what each reader checks: the files agree (the .bin's stages and the .const's size
// are the pilfflonkinfo's, and the rows of the .bin's constraints lie in the trace), and what this
// prover supports: a layout that packs every committed column once, and every piece of Q once, the
// pieces Q0 … Q<m−1> of cmPolsMap (Q0 alone if Q is not split, piece i at stageId and stagePos i),
// in f of their own opened at ξ, each f of the degree the pieces' bounds give it
// (pilfflonk/docs/protocol.md#degrees); and hints that are im_col, gsum_col and gprod_col only
// (im_airval computes an air value, pilfflonk/docs/README.md#scope), which give every column of
// stages 2 and above but the im pols, each once, from operands that read only what is computed
// before them (stdHints()), and update no airgroup value; im_col ones only in an AIR with a
// gsum_col or gprod_col, as the STARK's calculateImHints computes them only then.
class AirKey {
public:
    // From the files' contents. `name` is what the errors call the AIR. Throws FormatError. With a
    // `gpu` (only in a library built with the GPU), which must outlive the key, the key is on it: its
    // fixed columns are interpolated and committed on the device, where their coefficients stay, and
    // its proofs run there (GpuAirKey, pilfflonk_key_gpu.hpp). It throws then as GpuAirKey's
    // constructor too: std::invalid_argument if the device has not the memory of the key and a proof
    // of the AIR.
    AirKey(PilfflonkInfo info, ExpressionsBin bin, const uint8_t *constants, uint64_t constantsBytes,
           const std::string &name, GpuKey *gpu = nullptr);
    ~AirKey();

    // Reads <dir>/<name>.pilfflonkinfo.json, <dir>/<name>.bin and <dir>/<name>.const. Throws
    // IoError and FormatError, and as the constructor.
    static std::unique_ptr<AirKey> load(const std::string &dir, const std::string &name, GpuKey *gpu = nullptr);

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
#ifdef __USE_CUDA__
    // Its device side, if the key is on the GPU; null otherwise.
    const GpuAirKey *device() const { return deviceKey.get(); }
#endif

    // Fixed column c (constPolsMap index) on H, N values in natural order.
    const FrElement *fixedEvaluations(uint64_t c) const { return fixedEvals.get() + c * n(); }
    // Its interpolant, of N coefficients. Not const, as rapidsnark's API takes it, but never changed.
    Poly *fixedPolynomial(uint64_t c) const { return fixedPolys[c].get(); }

    // Where the layout commits constPolsMap[id] or cmPolsMap[id]; f = UINT64_MAX if it does not
    // (a column the evMap never opens, pilfflonk/docs/protocol.md#layout).
    const LayoutPosition &constPosition(uint64_t id) const { return constPositions[id]; }
    const LayoutPosition &cmPosition(uint64_t id) const { return cmPositions[id]; }
    static constexpr uint64_t NOT_COMMITTED = UINT64_MAX;

    // The coefficients of b(X) of the blinding of every column of f
    // (pilfflonk/docs/protocol.md#blinding): |O_f| + 1 for an f of a committed stage, 0 for a fixed
    // one and for Q's (whose pieces, if it is split, are blinded at their boundaries instead:
    // AirDegrees).
    uint64_t blindLength(uint64_t f) const;

    // The witness of an instance: the stagePos in stage 1 of each of its C columns, column c being
    // the cmPolsMap entry of stage 1 that is not an im pol and has stageId c
    // (pilfflonk/docs/formats.md#witness-directory).
    const std::vector<uint64_t> &witnessColumns() const { return witness; }

    // The cmPolsMap index of the column of stage s (1 … nStages + 1) at stagePos p: cmIds()[s][p].
    const std::vector<std::vector<uint64_t>> &cmIds() const { return cmIdsByStage; }

    // The columns the code of Q (cExpId) reads: fixed ones and committed ones of stages 1 … nStages,
    // each committed by the layout (checked when the key is built).
    const std::vector<ColumnRead> &qReads() const { return qColumns; }

    // The number of f of the layout of stage 0 (the fixed ones), which come first
    // (pilfflonk/docs/protocol.md#global-order).
    uint64_t nFixedF() const { return nFixed; }

    // The commitments [f(τ)]₁ of the fixed f, in the order of the layout, from the fixed columns of
    // the .const: their interpolants packed (pack()) and committed with `srs`, as the setup commits
    // them for the vkey (commitFixed; nothing is blinded). One MSM per f, of its k·N coefficients;
    // on the GPU, those the key computed on the device when it loaded, if `srs` is the SRS whose
    // powers it holds (GpuKey::holds).
    // The prover never needs them, the verifier takes them from the vkey: the orchestrator compares
    // them, so that a .const the vkey was not set up with is refused instead of giving proofs that
    // do not verify. Throws std::invalid_argument if an f has more coefficients than `srs` has
    // powers (a ProvingKey checks that its SRS has enough).
    std::vector<G1Point> fixedCommitments(const Srs &srs) const;

    // The number of pieces of Q, m (1 if it is not split), and where the layout commits piece i: its
    // f, of stage nStages + 1, and its index j in it.
    uint64_t nQPieces() const { return qPositions.size(); }
    const LayoutPosition &qPosition(uint64_t piece) const { return qPositions.at(piece); }

    // The im_col, gsum_col and gprod_col hints of the .bin, checked against the pilfflonkinfo: the
    // prover computes each column of stages 2 and above that is not an im pol from one of them. In
    // the order it computes them, the STARK's (StdHint::Kind): the im_col hints, then the gprod_col
    // ones and then the gsum_col ones, each in the .bin's order (getHintIdsByName). The operands of
    // a hint of stage s read fixed columns, columns of the stages before s, and of stage s only the
    // columns of the hints before it in this order: an im_col may read the im_col columns before it
    // (the std's product bus chains them), and a gsum_col or gprod_col those of the im_col hints.
    const std::vector<StdHint> &stdHints() const { return hints; }

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
    std::vector<LayoutPosition> qPositions; // by piece
    std::vector<StdHint> hints;
#ifdef __USE_CUDA__
    std::unique_ptr<GpuAirKey> deviceKey;
#endif
};

// Throws FormatError, naming the AIR, if an f of `air`'s layout has more coefficients than an SRS of
// nG1 powers [τ^i]₁.
void checkSrsFits(const AirKey &air, uint64_t nG1);

// The proving key of a proof (pilfflonk/docs/protocol.md#proof-sequence, step 1): the globalInfo,
// the SRS and every AIR's key, and, on the GPU, its device side (GpuKey). Immutable once built:
// proofs may share it, from several threads.
class ProvingKey {
public:
    // With a gpu, which must hold the SRS's powers [τ^i]₁ and be every AIR key's (AirKey's `gpu`).
    // Throws std::invalid_argument if an AIR key's is another, or the gpu holds another number of
    // points than the SRS.
    ProvingKey(GlobalInfo globalInfo, Srs srs, std::vector<std::vector<std::unique_ptr<AirKey>>> airs,
               std::shared_ptr<const GpuKey> gpu = nullptr);

    // Reads the provingKey/ at dir, as setup-pilfflonk writes it
    // (pilfflonk/docs/formats.md#provingkey):
    //   <dir>/pilout.globalInfo.json
    //   <dir>/<name>/pilfflonk/pilfflonk.srs.bin
    //   <dir>/<name>/<airgroup>/airs/<air>/air/<air>.{pilfflonkinfo.json, bin, const}
    // The vkey is not read: the digest the transcript absorbs is the orchestrator's
    // (pilfflonk/docs/protocol.md#transcript), which reads and checks the vkey. Throws IoError and
    // FormatError, and FormatError if an AIR's layout needs more powers [τ^i]₁ than the SRS holds
    // or its pilfflonkinfo is not the globalInfo's AIR. On Device::Gpu, the SRS's powers [τ^i]₁ are
    // copied to the GPU once they are read, each AIR's key is on it (AirKey's `gpu`), and its proofs
    // run there; the proofs are the same bit for bit.
    // Throws std::invalid_argument before it reads anything if there is no GPU (gpuAvailable()), in
    // a library built without one or on a machine without one, and as AirKey's constructor if the
    // device has not the memory of the key and a proof of each AIR.
    static std::unique_ptr<ProvingKey> load(const std::string &dir, Device device = Device::Cpu);

    ProvingKey(const ProvingKey &) = delete;
    ProvingKey &operator=(const ProvingKey &) = delete;

    const GlobalInfo &globalInfo() const { return info; }
    const Srs &srs() const { return structuredReferenceString; }
    Device device() const { return gpu ? Device::Gpu : Device::Cpu; }
    // Its device side, or null on the CPU.
    const GpuKey *gpuKey() const { return gpu.get(); }

    // The key of air airId of airgroup airgroupId. Throws std::invalid_argument if there is none.
    const AirKey &air(uint64_t airgroupId, uint64_t airId) const;

private:
    GlobalInfo info;
    // Before the SRS and the AIR keys, which point to it: it outlives them. A shared_ptr, whose deleter
    // is the GPU library's, so that a library built without the GPU never needs GpuKey's destructor.
    std::shared_ptr<const GpuKey> gpu;
    Srs structuredReferenceString;
    std::vector<std::vector<std::unique_ptr<AirKey>>> airKeys;
};

} // namespace PilFflonk

#endif
