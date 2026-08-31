// Round-trip tests of the STIR prover and verifier (Construction 5.2) with real Merkle trees, a
// real Fiat–Shamir transcript and real grinding. The polynomial arithmetic itself is covered by
// goldilocks/tests/test_stir_cpu.cpp; what is exercised here is the protocol: the transcript
// order, the coset/leaf conventions, the Merkle openings and the final consistency check.
#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <vector>

#include "stir.hpp"
#include "poseidon2_goldilocks.hpp"

namespace
{

using stir::E3;
using stir::FE;
using Stir = STIR<Goldilocks::Element>;
using Proof = StirProof<Goldilocks::Element>;

// The proof of work of the query messages, in the shape the STARK uses: permute
// [c_0, c_1, c_2, nonce, 0, 0, 0, 0] and require the first output limb below 2^{64−powBits}.
bool powOk(const uint64_t *challenge, uint64_t nonce, uint32_t powBits)
{
    if (powBits == 0) return true;   // 1 << 64 is UB below, and zero bits means no work to check

    Goldilocks::Element in[8], out[8];
    std::memset(in, 0, sizeof(in));
    std::memcpy(&in[0], challenge, FIELD_EXTENSION * sizeof(uint64_t));
    std::memcpy(&in[FIELD_EXTENSION], &nonce, sizeof(uint64_t));
    Poseidon2Goldilocks<8>::permute(out, in, Poseidon2Mode::Scalar);
    return Goldilocks::toU64(out[0]) < (uint64_t(1) << (64 - powBits));
}

void grind(uint64_t &nonce, const uint64_t *challenge, uint32_t powBits)
{
    for (nonce = 0; !powOk(challenge, nonce, powBits); nonce++)
    {
    }
}

StirParams testParams(bool hashCommits = false, uint64_t lastLevelVerification = 0)
{
    StirParams p;
    p.logFoldingFactors = {3, 3, 2};        // k_i
    p.logDegrees = {12, 9, 6, 4};           // d_i = d_{i−1} / k_{i−1}
    p.logDomainSizes = {14, 13, 12, 11};    // |L_{i+1}| = |L_i| / 2, initial rate 1/4
    p.numOodSamples = 1;                    // s
    p.numQueries = {12, 8, 6};              // t_i
    p.grindingBits = {2, 2, 2};
    p.merkleTreeArity = 4;
    p.lastLevelVerification = lastLevelVerification;
    p.merkleTreeCustom = true;
    p.transcriptArity = 4;
    p.hashCommits = hashCommits;
    return p;
}

// An honest f_0: the evaluations on L_0 of a random polynomial of degree < d_0.
void honestF0(std::vector<FE> &f0, const StirParams &params, uint64_t seed)
{
    std::mt19937_64 gen(seed);
    uint64_t d0 = uint64_t(1) << params.logDegrees[0];
    std::vector<FE> coeffs(d0 * FIELD_EXTENSION);
    for (uint64_t j = 0; j < coeffs.size(); j++) coeffs[j] = Goldilocks::fromU64(gen() % GOLDILOCKS_PRIME);

    stir::Domain L0{Stir::shift(), params.logDomainSizes[0]};
    f0.assign(L0.size() * FIELD_EXTENSION, Goldilocks::zero());
    stir::coefficientsToEvaluations(f0.data(), coeffs.data(), d0, L0);
}

const Goldilocks::Element SEED[4] = {Goldilocks::fromU64(1), Goldilocks::fromU64(2), Goldilocks::fromU64(3),
                                     Goldilocks::fromU64(4)};

TranscriptGL freshTranscript(const StirParams &params)
{
    TranscriptGL t(params.transcriptArity, params.merkleTreeCustom);
    t.put((Goldilocks::Element *)SEED, 4);
    return t;
}

// Prove, then verify. `checkF0` compares T_0's leaves against the true f_0, which is what the
// STARK verifier does by recomputing the DEEP quotient — and which pins down both the coset leaf
// layout and the member indexing of the round-1 queries.
bool roundTrip(const StirParams &params, std::vector<FE> &f0, Proof &proof, std::string &why)
{
    TranscriptGL tProve = freshTranscript(params);
    ProofTree<Goldilocks::Element> stageQueries(HASH_SIZE, params.numQueries[0], params.merkleTreeArity, params.lastLevelVerification);
    Stir::prove(proof, params, f0.data(), tProve, stageQueries, nullptr, 0, grind);

    TranscriptGL tVerify = freshTranscript(params);
    auto checkF0 = [&](uint64_t idx, const E3 &committed) {
        return stir::equal(committed, (const E3 &)f0[idx * FIELD_EXTENSION]);
    };
    return Stir::verify(proof, params, tVerify, checkF0, powOk, &why);
}

class StirProverTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite() { define_hash_family(HashFamily::Poseidon2); }
};

TEST_F(StirProverTest, honest_proof_verifies)
{
    StirParams params = testParams();
    std::vector<FE> f0;
    honestF0(f0, params, 0xA11CE);
    Proof proof = Stir::makeProof(params);
    std::string why;
    ASSERT_TRUE(roundTrip(params, f0, proof, why)) << why;

    // Every committed oracle was opened at its own query count.
    ASSERT_EQ(proof.trees.size(), params.M());
    for (uint64_t i = 0; i < params.M(); i++)
    {
        ASSERT_EQ(proof.trees[i].polQueries.size(), params.numQueries[i]) << "tree " << i;
    }
    ASSERT_EQ(proof.betas.size(), params.M() - 1);
    ASSERT_EQ(proof.finalPol.size(), (uint64_t(1) << params.logDegrees[params.M()]) * FIELD_EXTENSION);
}

TEST_F(StirProverTest, honest_proof_verifies_with_hashed_final_pol_and_last_levels)
{
    StirParams params = testParams(/*hashCommits=*/true, /*lastLevelVerification=*/2);
    std::vector<FE> f0;
    honestF0(f0, params, 0xB0B);
    Proof proof = Stir::makeProof(params);
    std::string why;
    ASSERT_TRUE(roundTrip(params, f0, proof, why)) << why;
}

// Every tampering below must be rejected. Each starts from the same honest proof.
class StirTamperTest : public StirProverTest
{
protected:
    StirParams params = testParams();
    std::vector<FE> f0;
    Proof honest = Stir::makeProof(params);

    void SetUp() override
    {
        honestF0(f0, params, 0xC0FFEE);
        std::string why;
        ASSERT_TRUE(roundTrip(params, f0, honest, why)) << why;
    }

    // Re-verify a (possibly tampered) copy of the honest proof.
    bool verifies(const Proof &proof, std::string &why)
    {
        TranscriptGL t = freshTranscript(params);
        auto checkF0 = [&](uint64_t idx, const E3 &committed) {
            return stir::equal(committed, (const E3 &)f0[idx * FIELD_EXTENSION]);
        };
        return Stir::verify(proof, params, t, checkF0, powOk, &why);
    }
};

TEST_F(StirTamperTest, corrupted_final_polynomial_is_rejected)
{
    Proof proof = honest;
    proof.finalPol[0] = proof.finalPol[0] + Goldilocks::one();
    std::string why;
    ASSERT_FALSE(verifies(proof, why));
}

TEST_F(StirTamperTest, corrupted_out_of_domain_answer_is_rejected)
{
    Proof proof = honest;
    proof.betas[0][0] = proof.betas[0][0] + Goldilocks::one();
    std::string why;
    ASSERT_FALSE(verifies(proof, why));
}

TEST_F(StirTamperTest, corrupted_leaf_of_the_initial_oracle_is_rejected)
{
    Proof proof = honest;
    proof.trees[0].polQueries[0][0].v[0][0] = proof.trees[0].polQueries[0][0].v[0][0] + Goldilocks::one();
    std::string why;
    ASSERT_FALSE(verifies(proof, why));
}

// The entry a round-1 query does *not* point at directly is still bound: it feeds the fold, and
// the Merkle path covers the whole leaf.
TEST_F(StirTamperTest, corrupted_non_representative_coset_member_is_rejected)
{
    Proof proof = honest;
    uint64_t k = uint64_t(1) << params.logFoldingFactors[0];
    ASSERT_GT(k, 1u);
    uint64_t last = (k - 1) * FIELD_EXTENSION;
    proof.trees[0].polQueries[0][0].v[last][0] = proof.trees[0].polQueries[0][0].v[last][0] + Goldilocks::one();
    std::string why;
    ASSERT_FALSE(verifies(proof, why));
}

TEST_F(StirTamperTest, corrupted_leaf_of_a_folded_oracle_is_rejected)
{
    Proof proof = honest;
    proof.trees[1].polQueries[0][0].v[0][0] = proof.trees[1].polQueries[0][0].v[0][0] + Goldilocks::one();
    std::string why;
    ASSERT_FALSE(verifies(proof, why));
}

TEST_F(StirTamperTest, corrupted_root_is_rejected)
{
    Proof proof = honest;
    proof.trees[1].root[0] = proof.trees[1].root[0] + Goldilocks::one();
    std::string why;
    ASSERT_FALSE(verifies(proof, why));
}

TEST_F(StirTamperTest, invalid_grinding_nonce_is_rejected)
{
    Proof proof = honest;
    proof.nonces[0] = proof.nonces[0] + 1;
    std::string why;
    ASSERT_FALSE(verifies(proof, why));
}

// Note: there is no "malicious prover" test here — `Prover::fold` asserts that an honest f_0 was
// given, so a dishonest one aborts rather than producing a proof. That the *arithmetic* rejects a
// high-degree f_0 is covered by STIR_TEST.dishonest_f0_is_not_low_degree_at_the_end.

// The shape a real air produces: fibonacci-square's Module air with lowDegreeTest = STIR and
// blake3 geometry — six iterations, a final folding factor of 1 (fold by two), arity 2, a
// published last level, and a hashed final polynomial. Grinding kept tiny so the test terminates.
TEST_F(StirProverTest, production_shaped_schedule)
{
    StirParams params;
    params.logFoldingFactors = {3, 3, 3, 3, 3, 1};
    params.logDegrees = {20, 17, 14, 11, 8, 5, 4};
    params.logDomainSizes = {21, 20, 19, 18, 17, 16, 15};
    params.numOodSamples = 1;
    params.numQueries = {211, 70, 43, 31, 24, 20};
    params.grindingBits = {2, 2, 2, 2, 2, 2};
    params.merkleTreeArity = 2;
    params.lastLevelVerification = 4;
    params.merkleTreeCustom = true;
    params.transcriptArity = 2;
    params.hashCommits = true;

    std::vector<FE> f0;
    honestF0(f0, params, 0xF1B0);
    Proof proof = Stir::makeProof(params);
    std::string why;
    ASSERT_TRUE(roundTrip(params, f0, proof, why)) << why;
}

} // namespace
