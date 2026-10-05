#ifndef PILFFLONK_TEST_HPP
#define PILFFLONK_TEST_HPP

// Shared by the pilfflonk tests, of the C API and of the C++ modules: plain asserts, one file per
// module, run from main() in pilfflonk_test.cpp, which also has the helpers below. Include this
// header first: it turns asserts on.
#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gmp.h>

#include "alt_bn128.hpp"
#ifdef __USE_CUDA__
#include "pilfflonk_kernels.hpp"
#include "pilfflonk_key_gpu.hpp"
#endif

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
void runPolynomialTests();
void runShplonkTests();
void runInfoTests();
void runExpressionsTests();
void runProverTests();
void runGpuTests();
void runExpressionsGpuTests();
void runHintsGpuTests();

// Whether the GPU path can be compared with the CPU one here: a library built with the GPU, on a
// machine with one. Without one it prints that `what` is skipped and returns false, or, with
// PILFFLONK_GPU=1 in the environment, fails: a GPU run must not pass by skipping.
bool gpuUnderTest(const char *what);

// The helpers of several tests, on ffiasm's arithmetic alone, not on the code under test.

// base^exponent, by ffiasm's square-and-multiply.
AltBn128::Engine::FrElement power(const AltBn128::Engine::FrElement &base, uint64_t exponent);
AltBn128::Engine::FrElement power(const AltBn128::Engine::FrElement &base, const mpz_t exponent);

AltBn128::Engine::FrElement fromUI(uint64_t value);

AltBn128::Engine::FrElement inverse(const AltBn128::Engine::FrElement &a);

// x·G for the generator G of G1: ffiasm's scalar multiplication, not its MSM.
AltBn128::Engine::G1Point g1Times(const AltBn128::Engine::FrElement &x);

bool samePoint(AltBn128::Engine::G1Point a, AltBn128::Engine::G1Point b);
// Through the projective comparison: ffiasm's mixed one does not compile warning-free.
bool samePoint(AltBn128::Engine::G1Point a, AltBn128::Engine::G1PointAffine b);

// The bytes of the file at `path`, which must open.
std::vector<uint8_t> readBytes(const std::string &path);

// `relative` in the repository, from $PILFFLONK_REPO_ROOT if it is set, and otherwise from the test
// binary's directory, pil2-stark/build or pil2-stark/build-gpu.
std::string repoPath(const std::string &relative);

#ifdef __USE_CUDA__
// `bytes` bytes at `data` in device memory of their own, one byte at least: an upload always has an
// address, which an operand's null pointer would not (a hint's number).
PilFflonk::DeviceBuffer upload(const void *data, uint64_t bytes);

template <typename T>
PilFflonk::DeviceBuffer upload(const std::vector<T> &host) {
    return upload(host.data(), host.size() * sizeof(T));
}

// The n values of type T at `device`, device memory, in host memory.
template <typename T = AltBn128::Engine::FrElement>
std::vector<T> download(const void *device, uint64_t n) {
    std::vector<T> host(n);
    gpu_plonk_memcpy_d2h(host.data(), device, n * sizeof(T));
    return host;
}

template <typename T>
std::vector<T> download(const PilFflonk::DeviceBuffer &device, uint64_t n) {
    return download<T>(device.data(), n);
}
#endif

} // namespace PilFflonkTest

#endif
