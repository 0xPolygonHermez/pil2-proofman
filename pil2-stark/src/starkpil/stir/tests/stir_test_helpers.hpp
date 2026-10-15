#ifndef STIR_TEST_HELPERS_HPP
#define STIR_TEST_HELPERS_HPP
// Fixtures shared by the CPU and GPU STIR tests, so both prove the same statement.
#include <cstring>
#include <random>
#include <vector>
#include "stir.hpp"
#include "poseidon2_goldilocks.hpp"

namespace stirtest
{
using stir::E3;
using stir::FE;
using Stir = STIR<Goldilocks::Element>;
using Proof = StirProof<Goldilocks::Element>;

// The proof of work of the query messages, in the shape the STARK uses: permute
// [c_0, c_1, c_2, nonce, 0, 0, 0, 0] and require the first output limb below 2^{64−grindingBits}.
inline bool powOk(const uint64_t *challenge, uint64_t nonce, uint32_t grindingBits)
{
    if (grindingBits == 0) return true;   // 1 << 64 is UB below, and zero bits means no work to check

    Goldilocks::Element in[8], out[8];
    std::memset(in, 0, sizeof(in));
    std::memcpy(&in[0], challenge, FIELD_EXTENSION * sizeof(uint64_t));
    std::memcpy(&in[FIELD_EXTENSION], &nonce, sizeof(uint64_t));
    Poseidon2Goldilocks<8>::permute(out, in, Poseidon2Mode::Scalar);
    return Goldilocks::toU64(out[0]) < (uint64_t(1) << (64 - grindingBits));
}

inline void grind(uint64_t &nonce, const uint64_t *challenge, uint32_t grindingBits)
{
    for (nonce = 0; !powOk(challenge, nonce, grindingBits); nonce++)
    {
    }
}

inline StirParams testParams(bool hashCommits = false, uint64_t lastLevelVerification = 0)
{
    StirParams p;
    p.logFoldingFactors = {3, 3, 2};        // k_i
    p.logDegrees = {12, 9, 6, 4};           // d_i = d_{i−1} / k_{i−1}
    p.logDomainSizes = {14, 13, 12, 11};    // |L_{i+1}| = |L_i| / 2, initial rate 1/4
    p.numOodSamples = 1;                    // s
    p.numQueries = {12, 8, 6};              // t_i
    p.grindingBitsQueries = {2, 2, 2};
    p.merkleTreeArity = 4;
    p.lastLevelVerification = lastLevelVerification;
    p.merkleTreeCustom = true;
    p.transcriptArity = 4;
    p.hashCommits = hashCommits;
    return p;
}

// An honest f_0: the evaluations on L_0 of a random polynomial of degree < d_0.
inline void honestF0(std::vector<FE> &f0, const StirParams &params, uint64_t seed)
{
    std::mt19937_64 gen(seed);
    uint64_t d0 = uint64_t(1) << params.logDegrees[0];
    std::vector<FE> coeffs(d0 * FIELD_EXTENSION);
    for (uint64_t j = 0; j < coeffs.size(); j++) coeffs[j] = Goldilocks::fromU64(gen() % GOLDILOCKS_PRIME);

    stir::Domain L0{Stir::shift(), params.logDomainSizes[0]};
    f0.assign(L0.size() * FIELD_EXTENSION, Goldilocks::zero());
    stir::coefficientsToEvaluations(f0.data(), coeffs.data(), d0, L0);
}

inline const Goldilocks::Element SEED[4] = {Goldilocks::fromU64(1), Goldilocks::fromU64(2), Goldilocks::fromU64(3),
                                     Goldilocks::fromU64(4)};

inline TranscriptGL freshTranscript(const StirParams &params)
{
    TranscriptGL t(params.transcriptArity, params.merkleTreeCustom);
    t.put((Goldilocks::Element *)SEED, 4);
    return t;
}

} // namespace stirtest
#endif
