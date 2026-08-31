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
#include <string>
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

// The prover's parameters, read off a stark info whose low-degree test is STIR.
inline StirParams stirParamsFromStarkInfo(const StarkInfo &starkInfo)
{
    const StirStruct &stir = starkInfo.starkStruct.stir;
    StirParams params;
    params.logFoldingFactors = stir.foldingFactors;
    params.logDegrees = stir.logDegrees;
    params.logDomainSizes = stir.logDomainSizes;
    params.numOodSamples = stir.numOodSamples;
    params.numQueries = stir.numQueries;
    params.grindingBits = stir.grindingBits;
    params.merkleTreeArity = starkInfo.starkStruct.merkleTreeArity;
    params.lastLevelVerification = starkInfo.starkStruct.lastLevelVerification;
    params.merkleTreeCustom = starkInfo.starkStruct.merkleTreeCustom;
    params.transcriptArity = starkInfo.starkStruct.transcriptArity;
    params.hashCommits = starkInfo.starkStruct.hashCommits;
    return params;
}

template <typename ElementType>
class STIR
{
public:
    using MerkleTreeType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, MerkleTreeGL, MerkleTreeBN128>;
    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;
    using Grinding = std::function<void(uint64_t &nonce, const uint64_t *challenge, uint32_t powBits)>;

    // Verifier-side hooks, so this file stays independent of the hash family and of the STARK.
    //
    // `F0Check` is called once per round-1 query with a uniform index of L_0 and the value T_0's
    // leaf claims for that point: f_0 is not the prover's to choose, it is the batched DEEP
    // polynomial, so the STARK verifier recomputes it there from the stage openings and compares.
    // Returning false rejects. May be empty in self-contained tests of the STIR argument alone.
    using F0Check = std::function<bool(uint64_t idxL0, const stir::E3 &committed)>;
    // Checks that `nonce` is a valid proof of work for `challenge` at `powBits` — the verifier's
    // side of `Grinding`. May be empty when `grindingBits` is all zeros.
    using GrindingCheck = std::function<bool(const uint64_t *challenge, uint64_t nonce, uint32_t powBits)>;

    // Run the prover on f_0, the evaluations of the batched DEEP polynomial on L_0 (extension
    // field, |L_0| = 2^{logDomainSizes[0]} elements). `transcript` is the STARK's transcript,
    // positioned right after the challenges that defined f_0. `stageTrees` are the committed
    // stage/constant/custom trees, opened at the representative point of each round-1 shift query.
    static void prove(StirProof<ElementType> &proof, const StirParams &params, const Goldilocks::Element *f0, TranscriptType &transcript, ProofTree<ElementType> &stageQueries, MerkleTreeType **stageTrees, uint64_t nStageTrees, const Grinding &grinding);

    // A proof sized for `params`, for callers with no StarkInfo at hand (`Proofs` owns one built
    // from the stark info instead).
    static StirProof<ElementType> makeProof(const StirParams &params)
    {
        StirProof<ElementType> proof;
        proof.init(params.numQueries, params.numOodSamples, params.logDegrees[params.M()], params.merkleTreeArity,
                   params.lastLevelVerification, std::is_same<ElementType, Goldilocks::Element>::value ? HASH_SIZE : 1);
        return proof;
    }

    // Verify a proof produced by `prove`. `transcript` must be in the same state `prove` received,
    // so that every challenge is re-derived rather than trusted. Returns false on the first failed
    // check, with a description in `failure` when given.
    static bool verify(const StirProof<ElementType> &proof, const StirParams &params, TranscriptType &transcript, const F0Check &checkF0, const GrindingCheck &checkGrinding, std::string *failure = nullptr);

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
void STIR<ElementType>::prove(StirProof<ElementType> &proof, const StirParams &params, const Goldilocks::Element *f0, TranscriptType &transcript, ProofTree<ElementType> &stageQueries, MerkleTreeType **stageTrees, uint64_t nStageTrees, const Grinding &grinding)
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

    // Sample the t_{i−1} shift queries of iteration i. As in FRI, a query is a uniform index of
    // L_{i−1}: its low bits give the coset — the leaf of T_{i−1}, equivalently the point of
    // L_{i−1}^{k_{i−1}} the query names — and its high bits give the member within that coset.
    //
    // Only round 1 uses the member part, and it matters there: f_0 has to be cross-checked against
    // the STARK's own commitments at a *uniform point of L_0*. Cross-checking the coset
    // representative instead would only ever touch the 1/k_0 of L_0 with member 0, leaving a prover
    // free to corrupt the other entries of every leaf.
    std::vector<uint64_t> round1RawIndices;
    auto sampleShiftQueries = [&](uint64_t i, std::vector<uint64_t> &cosetIndices) {
        E3 c;
        getChallenge(transcript, c);
        uint64_t nonce = 0;
        grinding(nonce, (const uint64_t *)&c[0], params.grindingBits[i - 1]);
        proof.nonces[i - 1] = nonce;

        TranscriptType transcriptQueries(params.transcriptArity, params.merkleTreeCustom);
        transcriptQueries.put(&c[0], FIELD_EXTENSION);
        transcriptQueries.put((Goldilocks::Element *)&nonce, 1);

        Domain Lprev = math.L(i - 1);
        uint64_t nLeaves = Lprev.size() >> params.logFoldingFactors[i - 1];   // = |L_{i−1}^{k_{i−1}}|
        std::vector<uint64_t> raw(params.numQueries[i - 1], 0);
        transcriptQueries.getPermutations(raw.data(), raw.size(), Lprev.logSize);

        cosetIndices.resize(raw.size());
        for (uint64_t j = 0; j < raw.size(); j++) cosetIndices[j] = raw[j] % nLeaves;
        if (i == 1) round1RawIndices = raw;
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

    // The round-1 queries also fix where f_0 is cross-checked against the STARK's own commitments:
    // as in the FRI prover, open the stage/constant/custom trees at each queried index of L_0, so
    // the verifier can recompute f_0 there and compare it against the matching member of T_0's leaf.
    {
        uint64_t maxBuffSize = 0;
        for (uint64_t t = 0; t < nStageTrees; t++)
        {
            maxBuffSize = std::max(maxBuffSize, stageTrees[t]->getMerkleTreeWidth() + stageTrees[t]->getMerkleProofSize());
        }
        std::vector<ElementType> buff(maxBuffSize);
        for (uint64_t j = 0; j < round1RawIndices.size(); j++)
        {
            std::vector<MerkleProof<ElementType>> openings;
            openings.reserve(nStageTrees);
            for (uint64_t t = 0; t < nStageTrees; t++)
            {
                openings.push_back(open(*stageTrees[t], round1RawIndices[j], buff.data())[0]);
            }
            stageQueries.polQueries[j] = openings;
        }
    }

    for (uint64_t i = 0; i < M; i++) delete trees[i];
}

template <typename ElementType>
bool STIR<ElementType>::verify(const StirProof<ElementType> &proof, const StirParams &params, TranscriptType &transcript, const F0Check &checkF0, const GrindingCheck &checkGrinding, std::string *failure)
{
    using namespace stir;

    auto fail = [&](const std::string &why) {
        if (failure != nullptr) *failure = why;
        return false;
    };

    const uint64_t M = params.M();
    if (M < 1 || params.numQueries.size() != M || params.grindingBits.size() != M) return fail("inconsistent parameters");
    if (proof.trees.size() != M || proof.nonces.size() != M) return fail("proof does not match the parameters");
    if (proof.betas.size() != M - 1) return fail("wrong number of out-of-domain answers");

    Parameters math{shift(), params.logFoldingFactors, params.logDegrees, params.logDomainSizes};
    math.validate();

    const uint64_t nFieldElements = std::is_same<ElementType, Goldilocks::Element>::value ? HASH_SIZE : 1;
    const uint64_t dFinal = uint64_t(1) << params.logDegrees[M];

    // ctx[i] is iteration i's (G_i, Ans_i, r_comb^i): what turns an opened value of the committed
    // g_i into the corresponding value of the virtual f_i. ctx[0] is unused — f_0 is committed
    // directly, not as a quotient.
    std::vector<QuotientContext> ctx(M);

    // Absorb into the transcript from a const proof.
    auto put = [&](const auto &v) {
        std::vector<typename std::decay_t<decltype(v)>::value_type> copy(v.begin(), v.end());
        transcript.put(copy.data(), copy.size());
    };

    // With a published bottom level, the root must be the reduction of that level: check it once
    // per tree, before anything is read out of it.
    if (params.lastLevelVerification > 0)
    {
        for (uint64_t i = 0; i < M; i++)
        {
            uint64_t nLeaves = math.L(i).size() >> params.logFoldingFactors[i];
            std::vector<ElementType> root = proof.trees[i].root, level = proof.trees[i].last_levels;
            if (!MerkleTreeType::verifyMerkleRoot(root.data(), level.data(), nLeaves, params.lastLevelVerification, params.merkleTreeArity, nFieldElements))
            {
                return fail("root of T_" + std::to_string(i) + " does not match its published last level");
            }
        }
    }

    // Re-derive iteration i's shift queries as uniform indices of L_{i−1} — the prover's
    // `sampleShiftQueries`, with the grinding *checked* instead of searched.
    auto deriveShiftQueries = [&](uint64_t i, std::vector<uint64_t> &raw) -> bool {
        E3 c;
        getChallenge(transcript, c);
        uint64_t nonce = proof.nonces[i - 1];
        if (checkGrinding && !checkGrinding((const uint64_t *)&c[0], nonce, params.grindingBits[i - 1])) return false;

        TranscriptType transcriptQueries(params.transcriptArity, params.merkleTreeCustom);
        transcriptQueries.put(&c[0], FIELD_EXTENSION);
        transcriptQueries.put((Goldilocks::Element *)&nonce, 1);
        raw.assign(params.numQueries[i - 1], 0);
        transcriptQueries.getPermutations(raw.data(), raw.size(), math.L(i - 1).logSize);
        return true;
    };

    // Read query q of iteration i out of T_{i−1} and recompute
    //   Ans_i(r_shift) = Fold(f_{i−1}, k_{i−1}, r^fold_{i−1})(r_shift).
    //
    // The leaf holds the k_{i−1} preimages of r_shift. For i = 1 those are values of f_0 directly;
    // for i ≥ 2 they are values of the *committed* g_{i−1}, and the virtual f_{i−1} is obtained
    // from them pointwise with ctx[i−1] — this is where a prover that committed a g_{i−1}
    // disagreeing with Ans_{i−1} produces a non-low-degree f_{i−1} and is caught downstream.
    auto foldAtQuery = [&](uint64_t i, uint64_t q, uint64_t raw, const E3 &rFold, E3 &out) -> bool {
        const ProofTree<ElementType> &tree = proof.trees[i - 1];
        const uint64_t logK = params.logFoldingFactors[i - 1];
        const uint64_t k = uint64_t(1) << logK;
        const Domain Lprev = math.L(i - 1);
        const uint64_t nLeaves = Lprev.size() >> logK;
        const uint64_t leaf = raw % nLeaves;

        if (q >= tree.polQueries.size() || tree.polQueries[q].size() != 1) return false;
        const MerkleProof<ElementType> &mkp = tree.polQueries[q][0];
        if (mkp.v.size() != k * FIELD_EXTENSION) return false;

        std::vector<Goldilocks::Element> values(k * FIELD_EXTENSION);
        for (uint64_t e = 0; e < values.size(); e++) values[e] = mkp.v[e][0];

        MerkleTreeType mt(params.merkleTreeArity, params.lastLevelVerification, params.merkleTreeCustom, nLeaves, k * FIELD_EXTENSION);
        std::vector<std::vector<ElementType>> siblings = mkp.mp;
        std::vector<ElementType> root = tree.root, level = tree.last_levels;
        if (level.empty()) level.resize(nFieldElements);   // unused when lastLevelVerification == 0
        if (!mt.verifyGroupProof(root.data(), level.data(), siblings, leaf, values)) return false;

        if (i == 1)
        {
            uint64_t member = raw / nLeaves;
            if (checkF0 && !checkF0(raw, (const E3 &)values[member * FIELD_EXTENSION])) return false;
        }
        else
        {
            for (uint64_t j = 0; j < k; j++)
            {
                E3 x, fx;
                embed(x, Lprev.point(leaf + j * nLeaves));
                ctx[i - 1].apply(fx, (const E3 &)values[j * FIELD_EXTENSION], x);
                Goldilocks3::copy((E3 &)values[j * FIELD_EXTENSION], fx);
            }
        }

        NTT_Goldilocks ntt(k, 1);
        std::vector<FE> scratch;
        foldCoset(out, values.data(), Lprev, logK, leaf, rFold, ntt, scratch);
        return true;
    };

    // ---- Initial commitment and the first folding challenge ------------------------------------
    put(proof.trees[0].root);
    E3 rFold;
    getChallenge(transcript, rFold);   // r^fold_0

    // ---- Main loop: iterations i = 1, …, M−1 --------------------------------------------------
    for (uint64_t i = 1; i < M; i++)
    {
        put(proof.trees[i].root);

        // 2(b)  r_out^{i,1..s} ← F \ L_i, drawn exactly as the prover drew them.
        std::vector<E3> rOut(params.numOodSamples);
        for (uint64_t j = 0; j < params.numOodSamples; j++)
        {
            do
            {
                getChallenge(transcript, rOut[j]);
            } while (math.L(i).contains(rOut[j]));
        }

        // 2(c)  the prover's β_{i,·}.
        if (proof.betas[i - 1].size() != params.numOodSamples * FIELD_EXTENSION) return fail("wrong β count in iteration " + std::to_string(i));
        put(proof.betas[i - 1]);

        // 2(d)  r^fold_i, r_comb^i, r_shift^{i,·}.
        E3 rFoldNext, rComb;
        getChallenge(transcript, rFoldNext);
        getChallenge(transcript, rComb);
        std::vector<uint64_t> raw;
        if (!deriveShiftQueries(i, raw)) return fail("invalid grinding in iteration " + std::to_string(i));

        // 2(e)  build (G_i, Ans_i): the out-of-domain claims, plus the fold values the verifier
        //       recomputes itself from T_{i−1}. No equality is checked here — the binding is what
        //       the quotient does to the next iteration's opened values.
        ctx[i].reset(rComb);
        for (uint64_t j = 0; j < params.numOodSamples; j++)
        {
            ctx[i].add(rOut[j], (const E3 &)proof.betas[i - 1][j * FIELD_EXTENSION]);
        }
        const Domain LprevK = math.L(i - 1).power(params.logFoldingFactors[i - 1]);
        for (uint64_t q = 0; q < raw.size(); q++)
        {
            E3 v;
            if (!foldAtQuery(i, q, raw[q], rFold, v)) return fail("query " + std::to_string(q) + " of iteration " + std::to_string(i) + " failed");
            E3 pt;
            embed(pt, LprevK.point(raw[q] % LprevK.size()));
            ctx[i].add(pt, v);
        }
        if (ctx[i].size() >= math.d(i)) return fail("|G| is not below d_i in iteration " + std::to_string(i));
        ctx[i].build();

        Goldilocks3::copy(rFold, rFoldNext);
    }

    // ---- Final step ---------------------------------------------------------------------------
    // p in the clear, then the only explicit equality check of the whole protocol:
    //   Fold(f_{M−1}, k_{M−1}, r^fold_{M−1})(r_shift) = p(r_shift).
    if (proof.finalPol.size() != dFinal * FIELD_EXTENSION) return fail("wrong size for the final polynomial");
    if (!params.hashCommits)
    {
        put(proof.finalPol);
    }
    else
    {
        TranscriptType transcriptHash(params.transcriptArity, params.merkleTreeCustom);
        std::vector<Goldilocks::Element> copy(proof.finalPol.begin(), proof.finalPol.end());
        transcriptHash.put(copy.data(), copy.size());
        ElementType hash[HASH_SIZE];
        transcriptHash.getState(hash);
        transcript.put(hash, HASH_SIZE);
    }

    std::vector<uint64_t> raw;
    if (!deriveShiftQueries(M, raw)) return fail("invalid grinding in the final round");
    const Domain LprevK = math.L(M - 1).power(params.logFoldingFactors[M - 1]);
    for (uint64_t q = 0; q < raw.size(); q++)
    {
        E3 v;
        if (!foldAtQuery(M, q, raw[q], rFold, v)) return fail("final query " + std::to_string(q) + " failed");
        E3 x, px;
        embed(x, LprevK.point(raw[q] % LprevK.size()));
        evalPoly(px, proof.finalPol.data(), dFinal, x);
        if (!equal(px, v)) return fail("final consistency check failed at query " + std::to_string(q));
    }

    return true;
}

#endif
