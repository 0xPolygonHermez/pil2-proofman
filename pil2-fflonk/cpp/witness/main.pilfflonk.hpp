#ifndef PIL2_FFLONK_WITNESS_HPP
#define PIL2_FFLONK_WITNESS_HPP

#include <string>

#include <alt_bn128.hpp>
#include <nlohmann/json.hpp>

// The circom witness calculator, declared but not ported.
//
// pil-fflonk's prover has two ways in: `prove(committedPolsFilename)` reads a
// committed trace, and `prove(execFilename, circomVerifier, ...)` generates one
// by running a circom circuit first. Only the first is wanted here -- the
// second belongs to the recursion flow, which pil2-proofman already has its own
// path for (recursivef bridges Goldilocks to BN128 and wraps the result).
//
// Declaring it keeps the prover compiling unedited; calling it says so rather
// than failing to link.
namespace CircomPilFflonk {

void getCommittedPols(AltBn128::Engine &E, AltBn128::FrElement *pAddress, const std::string &circomVerifier,
                      const std::string &execFile, const std::string &zkinFile, uint64_t nCols, uint64_t N);

void getCommittedPols(AltBn128::Engine &E, AltBn128::FrElement *pAddress, const std::string &circomVerifier,
                      const std::string &execFile, nlohmann::json &zkin, uint64_t nCols, uint64_t N);

} // namespace CircomPilFflonk

#endif // PIL2_FFLONK_WITNESS_HPP
