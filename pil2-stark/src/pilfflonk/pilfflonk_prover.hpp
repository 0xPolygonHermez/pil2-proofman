#ifndef PILFFLONK_PROVER_HPP
#define PILFFLONK_PROVER_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "alt_bn128.hpp"
#include "pilfflonk_commit.hpp"
#include "pilfflonk_proving_key.hpp"
#include "pilfflonk_rng.hpp"
#include "pilfflonk_shplonk_prover.hpp"
#include "pilfflonk_transcript.hpp"

namespace PilFflonk {

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

// The prover side of one instance of an AIR (spec §4.4, steps 2 and 3): its columns, their
// committed polynomials and Q. The orchestrator drives it in the order of the transcript (A.4):
// commitStage(1), …, commitStage(nStages), then commitQ; the challenges each step needs come from
// the transcript between them. An Opening then evaluates and opens the committed polynomials.
//
// commitStage(s):
//   1. the columns of stage s: the witness's for s = 1 (stages >= 2 come from the std's prover
//      hints, not supported yet: plan M30);
//   2. the intermediate polynomials of stage s, with the bytecode on H (ExpressionsDomain::trace):
//      each once the columns its code reads are, in any order that allows it;
//   3. for each f of stage s in the layout: the INTT of each column p_j with room for its blinding
//      (Lde::intt), the blinding p_j'(X) = p_j(X) + (X^N − 1)·b_j(X) with |O_f| + 1 coefficients of
//      b_j from the BlindingSource (spec A.3), in the order of the layout and of f's columns; then f
//      packed (pack(), spec A.2) and committed (Srs::commit).
// commitQ:
//   every column Q's code reads extended to the coset g·H' from its committed polynomial
//   (Lde::extendCoset, the blinding included), Q (cExpId) on the coset with the challenges and
//   the values, back to coefficients (Lde::interpolateCoset), and committed unblinded (spec A.3,
//   Q not split). Its coefficients from the bound of spec A.1 on must be zero: if not, the witness
//   does not satisfy the constraints, and commitQ throws UnsatisfiedError.
// check:
//   pilfflonk check (spec §4.4, "Depuració"; plan M25), which proves nothing: the im pols of stage 1
//   as commitStage(1) computes them, then the numerator of each constraint of the .bin (section 2)
//   on H, with the bytecode, and the rows of its domain where it is not 0.
//
// Elements are in Montgomery form. Refused arguments throw std::invalid_argument before anything
// changes. Not safe to use from several threads at once; the ProvingKey, which must outlive it, may
// be shared.
class Instance {
public:
    // Instance of air airId of airgroup airgroupId of pk. stage1 holds its witness as the witness
    // directory's .bin does (spec A.6): the N rows one after another, each the C stage-1 columns of
    // AirKey::witnessColumns() as canonical 32-byte little-endian scalars, N·C·32 bytes. airValues
    // and proofValues are the values of stage 1, in the order of their stage-1 entries of the AIR's
    // airValuesMap and of the globalInfo's proofValuesMap; publics, the globalInfo's nPublics.
    // Throws std::invalid_argument if there is no such AIR, if a count or the size is not the one
    // expected, or if a scalar of stage1 is not below r (naming its row and column).
    Instance(const ProvingKey &pk, uint64_t airgroupId, uint64_t airId, const uint8_t *stage1, uint64_t stage1Bytes,
             std::vector<FrElement> airValues, std::vector<FrElement> publics, std::vector<FrElement> proofValues,
             std::unique_ptr<BlindingSource> blinding);

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
    // at most nStages, and for a stage >= 2 (plan M30).
    std::vector<G1Point> commitStage(uint64_t stage, const std::vector<FrElement> &challenges);

    // Q, with the challenges of stage nStages + 1 (std_vc): the commitments of Q's f. Throws
    // std::invalid_argument unless every stage is committed and Q is not, and UnsatisfiedError (Q
    // stays uncommitted) if the witness does not satisfy the AIR's constraints.
    std::vector<G1Point> commitQ(const std::vector<FrElement> &challenges);

    // The check of the witness against every constraint of section 2 of the AIR's .bin, in its order
    // (the pilout's constraints, then those of the im pols, im − e): for each, the rows of its domain
    // where it does not hold, the first maxRows of them with the value there. It changes nothing
    // the commits depend on, and may run before, between or after them. Throws std::invalid_argument,
    // before computing anything, if a constraint is of a stage >= 2, whose columns come from the std's
    // prover hints (plan M30).
    std::vector<ConstraintCheck> check(uint64_t maxRows);

    // p_j of f (a non-fixed entry of the layout) once its stage is committed; null before. Not const
    // as rapidsnark's API takes it, but never changed.
    Poly *polynomial(uint64_t f, uint64_t j) const;

private:
    void setChallenges(uint64_t stage, const std::vector<FrElement> &given);
    ProverValues valuesOnTrace() const;
    void computeImPols(uint64_t stage);
    std::vector<G1Point> commitF(uint64_t stage);

    const ProvingKey &pk;
    const AirKey &key;
    std::unique_ptr<BlindingSource> blinding;
    uint64_t next = 1;
    // The im pols of stages 1 … imPolsComputed are in columns: computeImPols does each stage once.
    uint64_t imPolsComputed = 0;
    std::vector<FrElement> publicValues;
    std::vector<FrElement> airValueValues;
    std::vector<FrElement> proofValueValues;
    std::vector<FrElement> challengeValues; // challengesMap order
    // columns[s]: the columns of stage s (1 … nStages) on H, column p at [p·N, (p+1)·N).
    std::vector<std::vector<FrElement>> columns;
    // By cmPolsMap index: the committed polynomial of a column, and its buffer.
    std::vector<std::unique_ptr<FrElement[]>> coefBuffers;
    std::vector<std::unique_ptr<Poly>> polys;
    std::unique_ptr<Poly> q;
};

// The opening of a proof (spec §4.4, steps 4 and 5; A.5): every f of its instances in the global
// order of A.5, evaluated at the roots of ξ·ω^s for each of its offsets, and SHPLONK's W and W'.
// Built on M7's ShplonkProver, which computes the evaluations as it is built.
//
// The global order: the fixed f of each AIR with an instance, in canonical order of the AIRs, then
// the non-fixed f of each instance in canonical order, each in the order of its layout (Q last).
//
// Keeps pointers to the instances' polynomials: the instances, unchanged, must outlive it.
class Opening {
public:
    // The instances, in canonical order ((airgroupId, airId) non-decreasing), of one ProvingKey, all
    // with Q committed and of the same N (the one invZh is of), at ξ = xiSeed^powerW, powerW the lcm
    // of the k of every f (spec A.2.5). Throws std::invalid_argument if they are not, and
    // std::runtime_error if ξ is in H (probability N/r).
    Opening(const std::vector<const Instance *> &instances, const FrElement &xiSeed);

    // The evaluations of the proof, in the order of A.4 step 4 and of the proof (A.6): for each AIR
    // with an instance, those of its fixed columns, and then for each instance those of its other
    // columns, each in the order of the AIR's evMap: evMap entry (type, id, prime) is its column at
    // ξ·ω^prime.
    const std::vector<FrElement> &evaluations() const { return proofEvaluations; }

    // Q(ξ) of instance i: the value the verifier computes from the evaluations (spec A.1) and SHPLONK
    // opens Q's f at. For tests and diagnostics; not part of the proof.
    FrElement q(uint64_t instance) const;

    const FrElement &xi() const { return shplonk->xi(); }

    struct Proof {
        ShplonkProof shplonk; // α_S, [W]₁, y, [W']₁
        FrElement inv;        // verifierInverse(), see pilfflonk_shplonk_prover.hpp
        FrElement invZh;      // 1/Z_H(ξ) = 1/(ξ^N − 1)
    };

    // ShplonkProver::open on the transcript (spec A.4 step 5), which must hold everything absorbed
    // before it (the evaluations last), and the proof's inv and invZh.
    Proof open(Transcript &transcript) const;

private:
    const ProvingKey *pk = nullptr;
    uint64_t nBits = 0;
    std::unique_ptr<ShplonkProver> shplonk;
    std::vector<FrElement> proofEvaluations;
    std::vector<uint64_t> qGlobal; // the global index of each instance's Q f
};

} // namespace PilFflonk

#endif
