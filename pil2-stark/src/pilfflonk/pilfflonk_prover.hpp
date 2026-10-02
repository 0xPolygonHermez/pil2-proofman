#ifndef PILFFLONK_PROVER_HPP
#define PILFFLONK_PROVER_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "alt_bn128.hpp"
#include "pilfflonk_commit.hpp"
#include "pilfflonk_error.hpp"
#include "pilfflonk_proving_key.hpp"
#include "pilfflonk_rng.hpp"
#include "pilfflonk_shplonk_prover.hpp"
#include "pilfflonk_transcript.hpp"

namespace PilFflonk {

class InstanceGpu; // pilfflonk_instance_gpu.hpp
class OpeningGpu;  // pilfflonk_opening_gpu.hpp

// What Instance computes of the columns of a stage that the device path computes too
// (pilfflonk_hints_gpu.hpp), shared by both.

// The shift of a hint's column operand (HintInput::Kind::Column) on H, of N rows: on row i it reads
// row (i + shift) mod N, shift = offset mod N in [0, N).
uint64_t hintRowShift(const HintInput &in, uint64_t N);

// The im pols of stage `stage` of `air` (cmPolsMap indices), in the order Instance computes them:
// rounds over those not computed yet, in cmPolsMap order, each computed in the round where every
// column its code reads is (of an earlier stage, a witness column or hint column of its own, or an
// im pol computed before it). Throws FormatError, naming the first left, if they read each other in
// a cycle.
std::vector<uint64_t> imPolOrder(const AirKey &air, uint64_t stage);

// What Instance throws when the denominator of `hint` is 0 on row `row` of H, its first such row:
// the column has no value there.
UnsatisfiedError zeroDenominatorError(const AirKey &air, const StdHint &hint, uint64_t row);

// A row where a constraint does not hold: the value of its numerator there, not 0.
struct FailedRow {
    uint64_t row;
    FrElement value;
};

// What Instance::check finds of one constraint.
struct ConstraintCheck {
    // The rows firstRow <= i < lastRow of the constraint where its numerator is not 0.
    uint64_t nFailed = 0;
    // The first min(nFailed, maxRows) of them, in increasing order.
    std::vector<FailedRow> rows;
};

// The prover side of one instance of an AIR (pilfflonk/docs/protocol.md#proof-sequence, steps 2
// and 3): its columns, their committed polynomials and Q. The orchestrator drives it in the order
// of the transcript (pilfflonk/docs/protocol.md#transcript): commitStage(1), …,
// commitStage(nStages), then commitQ; the challenges each step needs come from the transcript
// between them. An Opening then evaluates and opens the committed polynomials.
//
// commitStage(s):
//   1. the columns of stage s: the witness's for s = 1; for s >= 2, those the std's prover hints give
//      (AirKey::stdHints, pilfflonk/docs/protocol.md#hint-columns) with the stage's challenges, in
//      the STARK's order (pil2-stark/src/starkpil/gen_proof.hpp), each on H with one batch
//      inversion:
//      - the im_col hints, as calculateImHints does (multiplyHintFields, hints.cpp): its reference
//        column is numerator/denominator on every row;
//      - then the gprod_col hints and the gsum_col ones, as calculateWitnessSTD does
//        (accMulHintFields): numerator_air/denominator_air on every row, accumulated row after row
//        into its reference column, a running product or a running sum. As in calculateWitnessSTD
//        with no airgroup value (v1 has none), result, numerator_direct and denominator_direct are
//        not read.
//      Each reads the columns of the hints before it (AirKey checked that it reads no other of the
//      stage). A denominator that is 0 on a row throws UnsatisfiedError: the column has no value
//      there;
//   2. the intermediate polynomials of stage s, with the bytecode on H (ExpressionsDomain::trace):
//      each once the columns its code reads are, in any order that allows it (the hints' columns
//      of stage s are, as step 1 computes them first);
//   3. for each f of stage s in the layout: the INTT of each column p_j with room for its blinding
//      (Lde::intt), the blinding p_j'(X) = p_j(X) + (X^N − 1)·b_j(X) with |O_f| + 1 coefficients of
//      b_j from the BlindingSource (pilfflonk/docs/protocol.md#blinding), in the order of the layout
//      and of f's columns; then f packed (pack(), pilfflonk/docs/protocol.md#layout) and committed
//      (Srs::commit).
// commitQ:
//   Q (cExpId) on the coset g·H' with the challenges and the values, one part of it at a time
//   (setQPartBits, pilfflonk/docs/protocol.md#q-in-parts): on each part, every column Q's code
//   reads extended to it from its committed polynomial (Lde::extendCosetPart, the blinding
//   included) and Q evaluated there (ExpressionsDomain::cosetPart); then Q back to coefficients
//   (Lde::interpolateCoset). Its coefficients from its bound (pilfflonk/docs/protocol.md#degrees)
//   on must be zero: if not, the witness does not satisfy the constraints, and commitQ throws
//   UnsatisfiedError. Then Q in its pieces (AirDegrees): Q itself, unblinded, if it is not split;
//   split, piece i its coefficients i·S … of Q, S = maxQDegree·N, and each boundary between pieces
//   i and i + 1 two factors b0, b1 from the BlindingSource, boundary by boundary and b0 first,
//   after the columns' ones: b0·X^S + b1·X^(S+1) added to piece i and b0 + b1·X subtracted from
//   piece i + 1, so that Σ_i X^(i·S)·Q_i(X) = Q(X) (pilfflonk/docs/protocol.md#q-pieces, as
//   pil-fflonk's pilfflonk_prover.cpp:697-720). Each f of Q's stage packs its pieces and is
//   committed.
// check:
//   pilfflonk check (pilfflonk/docs/README.md#pilfflonk-check), which proves nothing: the im pols
//   of stage 1 as commitStage(1) computes them, then the numerator of each constraint of the .bin
//   (section 2) on H, with the bytecode, and the rows of its domain where it is not 0. The columns
//   of the stages s >= 2 need their challenges, which check is given (pilfflonk check derives them
//   from fixed elements, as the STARK's verify-constraints does): it computes them as
//   commitStage(s) does, the hints' and then the im pols, into buffers of its own, and commits
//   nothing.
//
// On a key on the GPU (ProvingKey::load with Device::Gpu), the witness goes to the device as it is
// given, and every stage runs there (InstanceGpu): its hints and im pols, whose columns stay there
// (computeStageColumns), and the INTTs, the blinding and the commitments of commitStage, with the
// blinding factors drawn here; and so does commitQ whole: the extension of the columns to each part,
// Q's code there, its interpolation and the check of its bound, its pieces, their blinding with the
// factors drawn here, and their commitments. The committed polynomials and the pieces stay on the
// device for the opening, and the proof is the same bit for bit. Nothing comes back but counts, and
// what check, column, polynomial and qPiece ask for, copied on demand (decision D8). Such a key holds
// the device memory of one proof at a time: an instance holds it until it is destroyed, another
// thread's waits for it, and a second instance of this thread is refused.
//
// Elements are in Montgomery form. Refused arguments throw std::invalid_argument before anything
// changes. Not safe to use from several threads at once; the ProvingKey, which must outlive it, may
// be shared.
class Instance {
public:
    // Instance of air airId of airgroup airgroupId of pk. stage1 holds its witness as the witness
    // directory's .bin does (pilfflonk/docs/formats.md#witness-directory): the N rows one after
    // another, each the C stage-1 columns of AirKey::witnessColumns() as canonical 32-byte
    // little-endian scalars, N·C·32 bytes. airValues and proofValues are the values of stage 1, in
    // the order of their stage-1 entries of the AIR's airValuesMap and of the globalInfo's
    // proofValuesMap; publics, the globalInfo's nPublics. Throws std::invalid_argument if there is
    // no such AIR, if a count or the size is not the one expected, or if a scalar of stage1 is not
    // below r (naming its row and column), and, on a key on the GPU, if this thread has another
    // instance of it.
    Instance(const ProvingKey &pk, uint64_t airgroupId, uint64_t airId, const uint8_t *stage1, uint64_t stage1Bytes,
             std::vector<FrElement> airValues, std::vector<FrElement> publics, std::vector<FrElement> proofValues,
             std::unique_ptr<BlindingSource> blinding);
    ~Instance();

    Instance(const Instance &) = delete;
    Instance &operator=(const Instance &) = delete;

    const ProvingKey &provingKey() const { return pk; }
    const AirKey &air() const { return key; }
    uint64_t airgroupId() const { return key.info().airgroupId; }
    uint64_t airId() const { return key.info().airId; }

    // The stage commitStage takes next: 1 … nStages, then nStages + 1 for commitQ, then nStages + 2.
    uint64_t nextStage() const { return next; }
    bool qCommitted() const { return next > key.info().qStage(); }

    // The f of stage s (1 … nStages + 1) in the layout: how many commitments commitStage(s), or
    // commitQ for s = nStages + 1, returns.
    uint64_t nCommitments(uint64_t stage) const;

    // The number of challenges of stage s in the AIR's challengesMap.
    uint64_t nChallenges(uint64_t stage) const;

    // Stage `stage`, with its challenges (nChallenges(stage) of them, by stageId): the commitments of
    // its f, in the order of the layout. Throws std::invalid_argument unless stage is nextStage() and
    // at most nStages, and UnsatisfiedError (the stage stays uncommitted) if a hint's denominator is
    // 0 on a row.
    std::vector<G1Point> commitStage(uint64_t stage, const std::vector<FrElement> &challenges);

    // Q, with the challenges of stage nStages + 1 (std_vc): the commitments of the f of Q's pieces, in
    // the order of the layout. Throws std::invalid_argument unless every stage is committed and Q is
    // not, and UnsatisfiedError (Q stays uncommitted) if the witness does not satisfy the AIR's
    // constraints.
    std::vector<G1Point> commitQ(const std::vector<FrElement> &challenges);

    // The check of the witness against every constraint of section 2 of the AIR's .bin, in its order
    // (the pilout's constraints, then those of the im pols, im − e): for each, the rows of its domain
    // where it does not hold, the first maxRows of them with the value there. `challenges` are those
    // of stages 2 … nStages, by stage and then by stageId (none for an AIR of one stage), which the
    // columns of those stages are computed with (checkColumns). It changes nothing the commits depend
    // on, and may run before, between or after them; on a key on the GPU, whose witness columns are
    // on the device (decision D8), it copies them here the first time, and so only before commitQ,
    // which reuses their memory there: the first check after it throws std::invalid_argument. Throws
    // std::invalid_argument, before computing anything, if a constraint is of a stage the AIR does not
    // have or the number of challenges is not theirs, and UnsatisfiedError if a hint's denominator is
    // 0 on a row.
    std::vector<ConstraintCheck> check(uint64_t maxRows, const std::vector<FrElement> &challenges);

    // The columns of stages 1 … nStages on H that check checks, by stage then stagePos: stage 1's as
    // commitStage(1) computes them, and each later stage's as commitStage computes it, with
    // `challenges` (as check takes them), into new buffers. For tests and diagnostics; throws as
    // check does.
    std::vector<std::vector<FrElement>> checkColumns(const std::vector<FrElement> &challenges);

    // The N values on H of the column of stage `stage` at stagePos, as the prover computed them: the
    // witness's, a hint's or an im pol's. For tests and diagnostics (the oracle checks the hints'
    // columns against them); not part of the proof. Throws std::invalid_argument unless the stage is
    // committed and has such a column. On a key on the GPU, the column is on the device, which
    // computed it (decision D8): it is copied here the first time it is asked for, and so only before
    // commitQ, which reuses its memory there; asked for the first time after, it throws
    // std::invalid_argument.
    const FrElement *column(uint64_t stage, uint64_t stagePos) const;

    // p_j of f (a non-fixed entry of the layout) once its stage is committed; null before. Of Q's
    // stage, the piece of Q it packs. Not const as rapidsnark's API takes it, but never changed. On a
    // key on the GPU, which keeps the polynomials on the device, the first call for each copies it to
    // the host (for tests and diagnostics; qPiece for Q's).
    Poly *polynomial(uint64_t f, uint64_t j) const;

    // p_j of f as an Opening reads it: polynomial(f, j), but on a key on the GPU, where the device
    // keeps it (ShplonkComponent::elsewhere, of its coefficients and its degree), not copied to the
    // host.
    ShplonkComponent component(uint64_t f, uint64_t j) const;

    // Piece i of Q (the whole Q if it is not split) once Q is committed; null before. On a key on
    // the GPU, which keeps the pieces on the device, the first call copies them all to the host (for
    // tests and diagnostics).
    Poly *qPiece(uint64_t i) const;

    // How commitQ evaluates Q on the extended coset of N' = 2^nBitsExt points
    // (pilfflonk/docs/protocol.md#q-in-parts): in parts of 2^partBits points, one after another,
    // each the union of 2^(partBits − nBits) cosets of H, so that the columns Q reads are held on
    // one part at a time, 32·2^partBits bytes each, and not on all N'. By default partBits = nBits,
    // one coset of H per part, the least memory; nBitsExt evaluates Q on the whole coset at once.
    // Q, and so the proof, is the same bit for bit whatever the parts. Throws std::invalid_argument
    // unless nBits <= partBits <= nBitsExt, and, on a key on the GPU, if the key's device memory
    // cannot hold Q in such parts (InstanceGpu::requireQParts), saying how many bytes they need and
    // it has: the default parts, of 2^nBits points, it always holds.
    void setQPartBits(uint64_t partBits);

private:
    // The columns of stages 1 … nStages and the challenges (challengesMap order) check computes them with.
    struct CheckTrace {
        std::vector<std::vector<FrElement>> columns;
        std::vector<FrElement> challenges;
    };

    // The challenges of stage `stage` (by stageId) into `values`, of challengesMap order.
    void placeChallenges(uint64_t stage, const std::vector<FrElement> &given, std::vector<FrElement> &values) const;
    void setChallenges(uint64_t stage, const std::vector<FrElement> &given);
    // What the bytecode reads of `cols` (columns[s] as columns has them) and `challenges`.
    ProverValues valuesOn(const std::vector<std::vector<FrElement>> &cols,
                          const std::vector<FrElement> &challenges) const;
    // Its scalars alone, with `challenges`: no columns.
    ProverValues scalarValues(const std::vector<FrElement> &challenges) const;
    // The columns of stage `stage` its hints give, into cols[stage], with `challenges`.
    void computeHintColumns(uint64_t stage, std::vector<std::vector<FrElement>> &cols,
                            const std::vector<FrElement> &challenges) const;
    // The im pols of stage `stage`, into columns, once (imPolsComputed).
    void computeImPols(uint64_t stage);
    // The same, into cols[stage], with `challenges`.
    void computeImPols(uint64_t stage, std::vector<std::vector<FrElement>> &cols,
                       const std::vector<FrElement> &challenges) const;
    // Stage 1's im pols into columns, and, for an AIR of several stages, a copy of columns with the
    // later stages computed with `challenges` (empty for one stage).
    CheckTrace checkTrace(const std::vector<FrElement> &challenges);
    // On a key on the GPU, the witness columns into columns[1] from the device, once, before Q's
    // phase reuses their memory there: throws std::invalid_argument, naming `function`, if it has
    // begun. Nothing on the CPU, where they are there from the start.
    void witnessColumnsToHost(const char *function);
    // The blinding factors of the f of stage `stage`, b of them for each column of each
    // (blindLength), drawn f by f in the order of the layout and column by column within an f, one
    // BlindingSource::fill per column, in that order.
    std::vector<FrElement> drawBlinding(uint64_t stage);
    // The blinding factors of the boundaries between Q's pieces, two for each, b0 first, one
    // BlindingSource::fill per boundary, boundary by boundary: none if Q is not split.
    std::vector<FrElement> drawQBlinding();
    std::vector<G1Point> commitF(uint64_t stage);
    // What Q's code reads on a part of S points: column r of AirKey::qReads at columns + r·S (host or
    // device memory), and the instance's scalars, with the challenges of Q's stage.
    ProverValues qValuesOn(const FrElement *columns, uint64_t S) const;
    // The UnsatisfiedError of a Q with a coefficient of degree `degree`, at least its bound, not zero.
    UnsatisfiedError qAboveItsBound(uint64_t degree) const;
    // commitQ, here and on the device, in parts of 2^partBits points.
    std::vector<G1Point> commitQOnHost(uint64_t partBits);
#ifdef __USE_CUDA__
    std::vector<G1Point> commitQOnDevice(uint64_t partBits);
#endif

    const ProvingKey &pk;
    const AirKey &key;
    std::unique_ptr<BlindingSource> blinding;
    uint64_t next = 1;
    // setQPartBits; 0 is nBits.
    uint64_t qPartBits = 0;
    // The im pols of stages 1 … imPolsComputed are in columns: computeImPols does each stage once.
    uint64_t imPolsComputed = 0;
    std::vector<FrElement> publicValues;
    std::vector<FrElement> airValueValues;
    std::vector<FrElement> proofValueValues;
    std::vector<FrElement> challengeValues; // challengesMap order
    // columns[s]: the columns of stage s (1 … nStages) on H, column p at [p·N, (p+1)·N); on a key on
    // the GPU, which has them on the device, none, but for check's copy of the witness columns of
    // stage 1 (witnessColumnsToHost) and the im pols it computes from them.
    std::vector<std::vector<FrElement>> columns;
#ifdef __USE_CUDA__
    // Its device side, on a key on the GPU.
    std::unique_ptr<InstanceGpu> device;
    // commitQ has begun: its phase of the arena holds the bytes of the stages' columns from then on.
    bool qBegun = false;
    // The copies column() made of the device's columns of each stage (column p at p·N, once
    // copied[s][p]).
    mutable std::vector<std::vector<FrElement>> deviceCopies;
    mutable std::vector<std::vector<bool>> copied;
#endif
    // By cmPolsMap index: the committed polynomial of a column, and its buffer; on a key on the GPU,
    // the copies polynomial() made of the device's.
    mutable std::vector<std::unique_ptr<FrElement[]>> coefBuffers;
    mutable std::vector<std::unique_ptr<Poly>> polys;
    // By piece, once Q is committed; on a key on the GPU, once qPiece copies them, over qPieceCopy.
    mutable std::unique_ptr<FrElement[]> qPieceCopy;
    mutable std::vector<std::unique_ptr<Poly>> qPieces;
};

// The opening of a proof (pilfflonk/docs/protocol.md#proof-sequence, steps 4 and 5): every f of its
// instances in the global order (pilfflonk/docs/protocol.md#global-order), evaluated at the roots
// of ξ·ω^s for each of its offsets, and SHPLONK's W and W'. Built on ShplonkProver, which computes
// the evaluations as it is built.
//
// The global order: the fixed f of each AIR with an instance, in canonical order of the AIRs, then
// the non-fixed f of each instance in canonical order, each in the order of its layout (Q last).
//
// On a key on the GPU, whose one instance at a time holds its device memory, the evaluations, W, W'
// and their commitments are computed on the device (OpeningGpu) from the committed polynomials
// there, and the proof is the same bit for bit.
//
// Keeps pointers to the instances' polynomials: the instances, unchanged, must outlive it.
class Opening {
public:
    // The instances, in canonical order ((airgroupId, airId) non-decreasing), of one ProvingKey,
    // all with Q committed and of the same N (the one invZh is of), at ξ = xiSeed^powerW, powerW
    // the lcm of the k of every f (pilfflonk/docs/protocol.md#roots). Throws std::invalid_argument
    // if they are not, or if they are more than one of a key on the GPU (which holds one at a time:
    // the same instance twice), and std::runtime_error if ξ is in H (probability N/r).
    Opening(const std::vector<const Instance *> &instances, const FrElement &xiSeed);
    ~Opening();

    Opening(const Opening &) = delete;
    Opening &operator=(const Opening &) = delete;

    // The evaluations of the proof, in the order of the transcript and of the proof
    // (pilfflonk/docs/protocol.md#transcript, step 4; pilfflonk/docs/formats.md#proof): for each AIR
    // with an instance, those of its fixed columns, and then for each instance those of its other
    // columns, each in the order of the AIR's evMap: evMap entry (type, id, prime) is its column at
    // ξ·ω^prime. Then, for each instance whose Q is split, its pieces' Q_i(ξ), in the order of its
    // layout (its f of Q's stage, and each one's pieces in order).
    const std::vector<FrElement> &evaluations() const { return proofEvaluations; }

    // Q(ξ) of instance i: the value the verifier computes from the evaluations
    // (pilfflonk/docs/protocol.md#constraint-polynomial), and, split, Σ_i ξ^(i·S)·Q_i(ξ) of its
    // pieces, which it checks against it. For tests and diagnostics; not part of the proof.
    FrElement q(uint64_t instance) const;

    const FrElement &xi() const { return shplonk->xi(); }

    struct Proof {
        ShplonkProof shplonk; // α_S, [W]₁, y, [W']₁
        FrElement inv;        // verifierInverse(), see pilfflonk_shplonk_prover.hpp
        FrElement invZh;      // 1/Z_H(ξ) = 1/(ξ^N − 1)
    };

    // ShplonkProver::open on the transcript (pilfflonk/docs/protocol.md#transcript, step 5), which
    // must hold everything absorbed before it (the evaluations last), and the proof's inv and invZh.
    Proof open(Transcript &transcript) const;

private:
    const ProvingKey *pk = nullptr;
    uint64_t nBits = 0;
#ifdef __USE_CUDA__
    // Its device side, on a key on the GPU.
    std::unique_ptr<OpeningGpu> device;
#endif
    std::unique_ptr<ShplonkProver> shplonk;
    std::vector<FrElement> proofEvaluations;
    // Q(ξ) of each instance.
    std::vector<FrElement> qValues;
};

} // namespace PilFflonk

#endif
