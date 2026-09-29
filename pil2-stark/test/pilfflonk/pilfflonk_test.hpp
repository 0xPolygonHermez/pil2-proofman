#ifndef PILFFLONK_TEST_HPP
#define PILFFLONK_TEST_HPP

// Shared by the pilfflonk tests, of the C API and of the C++ modules: plain asserts, one file per
// module, run from main() in pilfflonk_test.cpp. Include this header first: it turns asserts on.
#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>

namespace PilFflonkTest {

// r and q written out independently of ffiasm, so that the tests also pin the moduli the library uses.
constexpr const char *R_HEX = "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";
constexpr const char *Q_HEX = "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47";

// A 256-bit integer as the C API passes it: 32 little-endian bytes.
struct Bytes32 {
    uint8_t bytes[32] = {};

    Bytes32() = default;

    // From a 64-digit big-endian hex string.
    explicit Bytes32(const char *hex) {
        assert(std::strlen(hex) == 64);
        for (int i = 0; i < 32; ++i) {
            bytes[31 - i] = static_cast<uint8_t>(std::stoul(std::string(hex + 2 * i, 2), nullptr, 16));
        }
    }

    bool operator==(const Bytes32 &other) const { return std::memcmp(bytes, other.bytes, sizeof(bytes)) == 0; }
    bool operator!=(const Bytes32 &other) const { return !(*this == other); }
};

void runTranscriptTests();
void runLdeTests();
void runSrsTests();
void runCommitTests();
void runShplonkTests();
void runInfoTests();
void runExpressionsTests();

} // namespace PilFflonkTest

#endif
