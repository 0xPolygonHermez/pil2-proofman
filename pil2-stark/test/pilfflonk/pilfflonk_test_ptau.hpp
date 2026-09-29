#ifndef PILFFLONK_TEST_PTAU_HPP
#define PILFFLONK_TEST_PTAU_HPP

// Test-only: a small, deterministic powers-of-tau file with a known τ (decision N13 of the plan),
// instead of downloading a ptau or running snarkjs. For the SRS and KZG tests, and for later ones
// that need an SRS whose τ they know (SHPLONK's identity, the prover).
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "alt_bn128.hpp"

namespace PilFflonkTest {

// A fresh directory for the files a test writes, next to the test binary: in the build directory,
// which git ignores, and apart from any other run. Removed, with the files file() named, when it
// goes out of scope; a failed assert leaves it behind to look at.
class TestDir {
public:
    TestDir();
    ~TestDir();
    TestDir(const TestDir &) = delete;
    TestDir &operator=(const TestDir &) = delete;

    const std::string &path() const { return dir; }

    // path()/name, removed with the directory.
    std::string file(const std::string &name);

private:
    std::string dir;
    std::vector<std::string> files;
};

// The τ of the test ptau: a fixed 253-bit scalar, so that every MSM sees full-width scalars.
AltBn128::Engine::FrElement testTau();

// The sections a test ptau is made of, as they go in the file (snarkjs's powersoftau_new.js):
// 1, the header (n8q = 32, q, power, ceremonyPower); 2, [τ^i]₁ for i < nG1; 3, [τ^i]₂ for i < nG2.
// Points are affine in ffiasm's Montgomery form, as snarkjs stores them. power is the smallest
// for which a real ptau would hold that many: 2^(power+1) - 1 >= nG1 and 2^power >= nG2.
struct PtauSections {
    uint32_t power;
    std::vector<uint8_t> header;
    std::vector<uint8_t> g1;
    std::vector<uint8_t> g2;
};
PtauSections testPtauSections(uint64_t nG1, uint64_t nG2, const AltBn128::Engine::FrElement &tau);

// Writes a binfile (rapidsnark's BinFileWriter) of `type` and `version` with these sections, in
// this order: well-formed or, for tests of the readers, not.
void writeBinFile(const std::string &path, const std::string &type, uint32_t version,
                  const std::vector<std::pair<uint32_t, std::vector<uint8_t>>> &sections);

// Writes a ptau of sections 1, 2 and 3 with τ = testTau(). A snarkjs ptau of the same power would
// also hold the rest of those powers and sections 4 to 7: this one holds only what pilfflonk
// reads, so it is quick to write.
void writeTestPtau(const std::string &path, uint64_t nG1, uint64_t nG2 = 2);

} // namespace PilFflonkTest

#endif
