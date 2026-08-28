#ifndef STIR_HPP
#define STIR_HPP

// The STIR prover (Arnon, Chiesa, Fenzi, Yogev — ePrint 2024/390), Construction 5.2.
//
// This file is the protocol driver: it owns the commitments, the transcript and the proof.
// The polynomial arithmetic (Fold, Quotient, DegCor, ĝ) lives in `stir_math.hpp`.
//
// Roles of the two sides, per iteration i = 1, …, M−1 (indices as in the paper):
//
//   P:  g_i := Fold(f_{i−1}, k_{i-1}, r^fold_{i−1}) on L_i                 → commit(g_i)          [2(a)]
//   V:  r_out^{i,1..s} ← F \ L_i                                                        [2(b)]
//   P:  β_{i,j} := ĝ_i(r_out^{i,j})                                                     [2(c)]
//   V:  r^fold_i, r_comb^i ← F;  r_shift^{i,1..t_{i−1}} ← L_{i−1}^{k_{i−1}}             [2(d)]
//   both:  G_i := {r_out^{i,·}} ∪ {r_shift^{i,·}},
//          Ans_i(r_out) := β,  Ans_i(r_shift) := Fold(f_{i−1}, k_{i-1}, r^fold_{i−1})(r_shift),
//          f_i := DegCor(d_i, r_comb^i, Quotient(g_i, G_i, Ans_i, Fill_i), d_i − |G_i|)   [2(e)]
//
// and the final step: P sends p := ĝ_M = Fold(f_{M−1}, k_{M-1}, r^fold_{M−1})^ in the clear, V samples
// r_shift^{M,1..t_{M−1}} ← L_{M−1}^{k_{M−1}} and checks Fold(f_{M−1}, k_{M−1}, r^fold_{M−1})(r_shift) = p(r_shift).
//
// What the verifier needs opened, and hence what the proof carries, is one coset of the oracle
// of f_{i−1} for every r_shift^{i,j}: the k_{i−1} preimages of r_shift, from which it recomputes
// Fold(f_{i−1}, k_{i-1}, r^fold_{i−1})(r_shift). f_{i−1} itself is a *virtual* oracle (a quotient of g_{i−1}),
// so what is committed and opened is g_{i−1}, and the verifier applies DegCor∘Quotient to the
// opened values with the G_{i−1}, Ans_{i−1} it already knows. For i = 1 the oracle is f_0, the
// batched DEEP polynomial; as in the FRI prover it is committed in cosets too (tree T_0) and the
// verifier cross-checks those leaves against the stage openings at one representative point.
//

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

#include "stir_math.hpp"
#include "proof_stark.hpp"
#include "merkleTreeGL.hpp"
#include "merkleTreeBN128.hpp"
#include "transcriptGL.hpp"
#include "transcriptBN128.hpp"

// Parameters of one STIR execution, as recorded in the stark info's `StirStruct` plus the
// solved security parameters.
struct StirParams
{
    std::vector<uint64_t> logFoldingFactors;   // k_i in bits, length M
    std::vector<uint64_t> logDegrees;          // log2 d_i, length M+1
    std::vector<uint64_t> logDomainSizes;      // log2 |L_i|, length M+1
    uint64_t numOodSamples;                    // s
    std::vector<uint64_t> numQueries;          // t_i, length M
    std::vector<uint64_t> grindingBits;        // grinding on iteration i+1's query message, length M

    // Commitment geometry, shared with the rest of the STARK.
    uint64_t merkleTreeArity;
    uint64_t lastLevelVerification;
    bool merkleTreeCustom;
    uint64_t transcriptArity;
    bool hashCommits;   // absorb hash(p) instead of p into the transcript

    uint64_t M() const { return logFoldingFactors.size(); }
};

// The proof of one STIR execution.
template <typename ElementType>
struct StirProof
{
    // trees[i] is the commitment to the oracle of f_i, for i = 0..M−1: T_0 commits f_0 and T_i
    // commits g_i (i ≥ 1), each in k_i-cosets. trees[i].polQueries are its t_i openings at the
    // shift queries r_shift^{i+1,·} of the next iteration.
    std::vector<ProofTree<ElementType>> trees;
    // betas[i−1] = β_{i,1..s}, the out-of-domain answers of iteration i = 1..M−1.
    std::vector<std::vector<Goldilocks::Element>> betas;
    // nonces[i−1]: grinding nonce of iteration i's query message, i = 1..M.
    std::vector<uint64_t> nonces;
    // Openings of the STARK's stage/constant/custom trees at the representative point of every
    // r_shift^{1,j}, from which the verifier recomputes f_0 there (as the FRI prover's `trees`).
    std::vector<std::vector<MerkleProof<ElementType>>> stageQueries;
    // p: the d_M coefficients of the final polynomial ĝ_M, in the clear.
    std::vector<Goldilocks::Element> finalPol;

    StirProof(const StirParams &params)
        : betas(params.M() >= 1 ? params.M() - 1 : 0, std::vector<Goldilocks::Element>(params.numOodSamples * FIELD_EXTENSION)),
          nonces(params.M(), 0),
          stageQueries(params.numQueries.empty() ? 0 : params.numQueries[0]),
          finalPol((uint64_t(1) << params.logDegrees[params.M()]) * FIELD_EXTENSION)
    {
        uint64_t nFieldElements = std::is_same<ElementType, Goldilocks::Element>::value ? HASH_SIZE : 1;
        for (uint64_t i = 0; i < params.M(); i++)
        {
            trees.emplace_back(nFieldElements, params.numQueries[i], params.merkleTreeArity, params.lastLevelVerification);
        }
    }
};

template <typename ElementType>
class STIR
{
public:
    using MerkleTreeType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, MerkleTreeGL, MerkleTreeBN128>;
    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;
    using Grinding = std::function<void(uint64_t &nonce, const uint64_t *challenge, uint32_t powBits)>;

    // Run the prover on f_0, the evaluations of the batched DEEP polynomial on L_0 (extension
    // field, |L_0| = 2^{logDomainSizes[0]} elements). `transcript` is the STARK's transcript,
    // positioned right after the challenges that defined f_0. `stageTrees` are the committed
    // stage/constant/custom trees, opened at the representative point of each round-1 shift query.
    static void prove(StirProof<ElementType> &proof, const StirParams &params, const Goldilocks::Element *f0, TranscriptType &transcript, MerkleTreeType **stageTrees, uint64_t nStageTrees, const Grinding &grinding);

    // The coset shift of every L_i: the same one the STARK's extended domain uses.
    static Goldilocks::Element shift() { return Goldilocks::shift(); }

private:
    // Commit the evaluations of an oracle on L_i in k_i-cosets: leaf m holds the k_i preimages of
    // the m-th point of L_i^{k_i}, i.e. the points of L_i at indices m + j·|L_i|/k_i, j = 0..k_i−1.
    static void commit(MerkleTreeType &tree, ProofTree<ElementType> &treeProof, const Goldilocks::Element *evals, const stir::Domain &L, uint64_t logK);
    // Open a committed oracle at leaf `idx`.
    static std::vector<MerkleProof<ElementType>> open(MerkleTreeType &tree, uint64_t idx, ElementType *buff);

    static void getChallenge(TranscriptType &transcript, stir::E3 &out) { transcript.getField((uint64_t *)&out[0]); }
};

template <typename ElementType>
void STIR<ElementType>::commit(MerkleTreeType &tree, ProofTree<ElementType> &treeProof, const Goldilocks::Element *evals, const stir::Domain &L, uint64_t logK)
{
    uint64_t k = uint64_t(1) << logK;
    uint64_t nLeaves = L.size() >> logK;   // = |L^k|
    assert(tree.height == nLeaves && tree.width == k * FIELD_EXTENSION);

    // Re-organise in cosets: leaf m ← f at indices m, m + nLeaves, m + 2·nLeaves, …
#pragma omp parallel for
    for (uint64_t m = 0; m < nLeaves; m++)
    {
        for (uint64_t j = 0; j < k; j++)
        {
            std::memcpy(&tree.source[(m * k + j) * FIELD_EXTENSION], &evals[(m + j * nLeaves) * FIELD_EXTENSION], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        }
    }
    tree.merkelize();
    tree.getRoot(&treeProof.root[0]);
    if (treeProof.last_level > 0) tree.getLevel(&treeProof.last_levels[0]);
}

template <typename ElementType>
std::vector<MerkleProof<ElementType>> STIR<ElementType>::open(MerkleTreeType &tree, uint64_t idx, ElementType *buff)
{
    tree.getGroupProof(&buff[0], idx);
    MerkleProof<ElementType> mkProof(tree.getMerkleTreeWidth(), tree.getMerkleProofLength(), tree.getNumSiblings(), &buff[0]);
    return std::vector<MerkleProof<ElementType>>{mkProof};
}

template <typename ElementType>
void STIR<ElementType>::prove(StirProof<ElementType> &proof, const StirParams &params, const Goldilocks::Element *f0, TranscriptType &transcript, MerkleTreeType **stageTrees, uint64_t nStageTrees, const Grinding &grinding)
{
    using namespace stir;

    const uint64_t M = params.M();
    assert(M >= 1);
    assert(params.numQueries.size() == M && params.grindingBits.size() == M);

    Parameters math{shift(), params.logFoldingFactors, params.logDegrees, params.logDomainSizes};
    Prover prover(math, f0);

    // The committed oracles: T_i commits f_0 (i = 0) or g_i (i ≥ 1) on L_i in k_i-cosets.
    std::vector<MerkleTreeType *> trees(M);
    for (uint64_t i = 0; i < M; i++)
    {
        Domain Li = math.L(i);
        trees[i] = new MerkleTreeType(params.merkleTreeArity, params.lastLevelVerification, params.merkleTreeCustom, Li.size() >> params.logFoldingFactors[i], math.k(i) * FIELD_EXTENSION, true, true);
    }

    // Sample the t shift queries of iteration i into L_{i−1}^{k_{i−1}} (t = t_{i−1}).
    // The round-1 indices are kept: they also fix where f_0 is cross-checked against the STARK's
    // own commitments (see the end of `prove`).
    std::vector<uint64_t> round1ShiftIndices;
    auto sampleShiftQueries = [&](uint64_t i, std::vector<uint64_t> &indices) {
        E3 c;
        getChallenge(transcript, c);
        uint64_t nonce = 0;
        grinding(nonce, (const uint64_t *)&c[0], params.grindingBits[i - 1]);
        proof.nonces[i - 1] = nonce;

        TranscriptType transcriptQueries(params.transcriptArity, params.merkleTreeCustom);
        transcriptQueries.put(&c[0], FIELD_EXTENSION);
        transcriptQueries.put((Goldilocks::Element *)&nonce, 1);
        indices.assign(params.numQueries[i - 1], 0);
        Domain LprevK = math.L(i - 1).power(params.logFoldingFactors[i - 1]);
        transcriptQueries.getPermutations(indices.data(), indices.size(), LprevK.logSize);
        if (i == 1) round1ShiftIndices = indices;
    };

    // Open T_{i−1} at every r_shift^{i,j}: the k_{i−1} preimages of r_shift are exactly leaf
    // `index` of T_{i−1} (see `commit`).
    auto openShiftQueries = [&](uint64_t i, const std::vector<uint64_t> &indices) {
        MerkleTreeType &tree = *trees[i - 1];
        std::vector<ElementType> buff(tree.getMerkleTreeWidth() + tree.getMerkleProofSize());
        for (uint64_t j = 0; j < indices.size(); j++)
        {
            proof.trees[i - 1].polQueries[j] = open(tree, indices[j], buff.data());
        }
    };

    // ---- Initial commitment and the first folding challenge ------------------------------------
    // f_0 is committed in k_0-cosets; the verifier reads f_0 through this tree.
    commit(*trees[0], proof.trees[0], f0, math.L(0), params.logFoldingFactors[0]);
    transcript.put(&proof.trees[0].root[0], proof.trees[0].root.size());

    E3 rFold;
    getChallenge(transcript, rFold);   // r^fold_0

    // ---- Main loop: iterations i = 1, …, M−1 --------------------------------------------------
    for (uint64_t i = 1; i < M; i++)
    {
        // 2(a)  g_i := Fold(f_{i−1}, k_{i-1}, r^fold_{i−1}), committed on L_i.
        prover.fold(rFold);
        commit(*trees[i], proof.trees[i], prover.g_next().data(), math.L(i), params.logFoldingFactors[i]);
        transcript.put(&proof.trees[i].root[0], proof.trees[i].root.size());

        // 2(b)  r_out^{i,1..s} ← F \ L_i. A squeezed element lies in L_i with probability
        //       |L_i| / |F| ≈ 2^{−170}; we still honour the definition by re-squeezing.
        std::vector<E3> rOut(params.numOodSamples);
        for (uint64_t j = 0; j < params.numOodSamples; j++)
        {
            do
            {
                getChallenge(transcript, rOut[j]);
            } while (math.L(i).contains(rOut[j]));
        }

        // 2(c)  β_{i,j} := ĝ_i(r_out^{i,j}).
        std::vector<E3> beta(params.numOodSamples);
        for (uint64_t j = 0; j < params.numOodSamples; j++)
        {
            prover.outOfDomainAnswer(beta[j], rOut[j]);
            std::memcpy(&proof.betas[i - 1][j * FIELD_EXTENSION], &beta[j][0], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        }
        transcript.put(&proof.betas[i - 1][0], proof.betas[i - 1].size());

        // 2(d)  r^fold_i, r_comb^i ← F and r_shift^{i,1..t_{i−1}} ← L_{i−1}^{k_{i−1}}.
        E3 rFoldNext, rComb;
        getChallenge(transcript, rFoldNext);
        getChallenge(transcript, rComb);
        std::vector<uint64_t> shiftIndices;
        sampleShiftQueries(i, shiftIndices);
        openShiftQueries(i, shiftIndices);

        // 2(e)  f_i := DegCor(d_i, r_comb^i, Quotient(g_i, G_i, Ans_i, Fill_i), d_i − |G_i|) on L_i,
        //       with Ans_i(r_shift) = Fold(f_{i−1}, k_{i-1}, r^fold_{i−1})(r_shift) taken from the fold.
        prover.degreeCorrect(rOut, beta, shiftIndices, rComb);

        Goldilocks3::copy(rFold, rFoldNext);
    }

    // ---- Final step ---------------------------------------------------------------------------
    // p := ĝ_M = Fold(f_{M−1}, k_{M-1}, r^fold_{M−1}), sent in the clear as its d_M coefficients.
    prover.fold(rFold);
    const std::vector<Goldilocks::Element> &p = prover.gHat_next();
    assert(p.size() == proof.finalPol.size());
    std::memcpy(proof.finalPol.data(), p.data(), p.size() * sizeof(Goldilocks::Element));
    if (!params.hashCommits)
    {
        transcript.put(proof.finalPol.data(), proof.finalPol.size());
    }
    else
    {
        TranscriptType transcriptHash(params.transcriptArity, params.merkleTreeCustom);
        transcriptHash.put(proof.finalPol.data(), proof.finalPol.size());
        ElementType hash[HASH_SIZE];
        transcriptHash.getState(hash);
        transcript.put(hash, HASH_SIZE);
    }

    // r_shift^{M,1..t_{M−1}} ← L_{M−1}^{k_{M−1}}: the verifier checks Fold(f_{M−1}, k_{M-1}, r^fold_{M−1})(r_shift)
    // = p(r_shift), so it needs the cosets of f_{M−1}'s oracle at these points.
    std::vector<uint64_t> finalShiftIndices;
    sampleShiftQueries(M, finalShiftIndices);
    openShiftQueries(M, finalShiftIndices);

    // The round-1 shift queries also fix where f_0 is cross-checked against the STARK's own
    // commitments: as in the FRI prover, open the stage/constant/custom trees at one representative
    // point of each queried coset of L_0 — the coset member with j = 0, whose index in L_0 is the
    // leaf index in T_0 — so the verifier can recompute f_0 there and compare it with T_0's leaf.
    {
        uint64_t maxBuffSize = 0;
        for (uint64_t t = 0; t < nStageTrees; t++)
        {
            maxBuffSize = std::max(maxBuffSize, stageTrees[t]->getMerkleTreeWidth() + stageTrees[t]->getMerkleProofSize());
        }
        std::vector<ElementType> buff(maxBuffSize);
        for (uint64_t j = 0; j < round1ShiftIndices.size(); j++)
        {
            std::vector<MerkleProof<ElementType>> openings;
            openings.reserve(nStageTrees);
            for (uint64_t t = 0; t < nStageTrees; t++)
            {
                openings.push_back(open(*stageTrees[t], round1ShiftIndices[j], buff.data())[0]);
            }
            proof.stageQueries[j] = openings;
        }
    }

    for (uint64_t i = 0; i < M; i++) delete trees[i];
}

#endif
