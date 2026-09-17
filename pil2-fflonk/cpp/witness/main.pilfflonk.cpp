#include "main.pilfflonk.hpp"

#include <stdexcept>

namespace CircomPilFflonk {

// See the header: the circom path is not ported, so say so at the call rather
// than at the link.
static void notPorted() {
    throw std::runtime_error(
        "pil2-fflonk: the circom witness path is not ported. Use prove(committedPolsFilename) with a committed "
        "trace; generating one from a circom circuit belongs to pil2-proofman's own recursion flow.");
}

void getCommittedPols(AltBn128::Engine &, AltBn128::FrElement *, const std::string &, const std::string &,
                      const std::string &, uint64_t, uint64_t) {
    notPorted();
}

void getCommittedPols(AltBn128::Engine &, AltBn128::FrElement *, const std::string &, const std::string &,
                      nlohmann::json &, uint64_t, uint64_t) {
    notPorted();
}

} // namespace CircomPilFflonk
