// Tests for the pilfflonk C API, the helpers the tests share (pilfflonk_test.hpp), and main(), which
// runs every test. Run with `make pilfflonk_test`, and against the GPU archive with
// `make pilfflonk_gpu_test` (pilfflonk/docs/README.md#tests); exits non-zero on the first failure.
#include "pilfflonk_test.hpp"

#include <limits.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>

#include "pilfflonk_api.hpp"
#include "pilfflonk_fr.hpp"

namespace PilFflonkTest {

namespace {

using Engine = AltBn128::Engine;

Engine &E = Engine::engine;

} // namespace

Engine::FrElement power(const Engine::FrElement &base, uint64_t exponent) {
    uint8_t bytes[sizeof(exponent)];
    for (size_t i = 0; i < sizeof(exponent); ++i) bytes[i] = static_cast<uint8_t>(exponent >> (8 * i));
    Engine::FrElement r;
    E.fr.exp(r, base, bytes, sizeof(bytes));
    return r;
}

Engine::FrElement power(const Engine::FrElement &base, const mpz_t exponent) {
    uint8_t littleEndian[32] = {};
    assert(mpz_sizeinbase(exponent, 256) <= sizeof(littleEndian));
    mpz_export(littleEndian, nullptr, -1, 1, -1, 0, exponent);
    Engine::FrElement result;
    E.fr.exp(result, base, littleEndian, sizeof(littleEndian));
    return result;
}

Engine::FrElement fromUI(uint64_t value) {
    Engine::FrElement e;
    E.fr.fromUI(e, value);
    return e;
}

Engine::FrElement inverse(const Engine::FrElement &a) {
    Engine::FrElement r;
    E.fr.inv(r, a);
    return r;
}

Engine::G1Point g1Times(const Engine::FrElement &x) {
    Engine::FrElement canonical;
    E.fr.fromMontgomery(canonical, x);
    Engine::G1Point p;
    E.g1.mulByScalar(p, E.g1.oneAffine(), reinterpret_cast<uint8_t *>(canonical.v), sizeof(canonical.v));
    return p;
}

bool samePoint(Engine::G1Point a, Engine::G1Point b) {
    return E.g1.eq(a, b);
}

bool samePoint(Engine::G1Point a, Engine::G1PointAffine b) {
    Engine::G1Point projective;
    E.g1.copy(projective, b);
    return E.g1.eq(a, projective);
}

std::vector<uint8_t> readBytes(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    assert(file);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::string repoPath(const std::string &relative) {
    if (const char *root = std::getenv("PILFFLONK_REPO_ROOT")) {
        return std::string(root) + "/" + relative;
    }
    char exe[PATH_MAX];
    const ssize_t length = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    assert(length > 0);
    exe[length] = '\0';
    const std::string dir(exe);
    return dir.substr(0, dir.rfind('/')) + "/../../" + relative;
}

#ifdef __USE_CUDA__
PilFflonk::DeviceBuffer upload(const void *data, uint64_t bytes) {
    PilFflonk::DeviceBuffer device(std::max<uint64_t>(bytes, 1));
    gpu_plonk_memcpy_h2d(device.data(), data, bytes);
    return device;
}
#endif

} // namespace PilFflonkTest

using PilFflonkTest::Bytes32;
using PilFflonkTest::R_HEX;

namespace {

// What the C API checks every scalar it is given with (decodeFr, firstNonCanonicalFr): below r.
void expectCanonical(const char *hex) {
    const Bytes32 s(hex);
    assert(PilFflonk::isCanonicalFr(s.bytes));
}

void expectNonCanonical(const char *hex) {
    const Bytes32 s(hex);
    assert(!PilFflonk::isCanonicalFr(s.bytes));
}

void testCheckCanonical() {
    expectCanonical("0000000000000000000000000000000000000000000000000000000000000000");
    expectCanonical("0000000000000000000000000000000000000000000000000000000000000001");
    expectCanonical("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000"); // r - 1
    // The top limbs match r and a lower one is smaller: the rest must not matter.
    expectCanonical("30644e72e131a029b85045b68181585d2833e84879b97090ffffffffffffffff");

    expectNonCanonical(R_HEX);
    expectNonCanonical("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000002"); // r + 1
    // The top limbs match r and a lower one is larger: the rest must not matter.
    expectNonCanonical("30644e72e131a029b85045b68181585d2833e84879b970920000000000000000");
    expectNonCanonical("3100000000000000000000000000000000000000000000000000000000000000");
    expectNonCanonical("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
}

void testSuccessClearsLastError() {
    uint8_t out[32];
    assert(pilfflonk_keccak256(nullptr, 1, out) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_last_error()[0] != '\0' && pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_keccak256(nullptr, 0, out) == PILFFLONK_OK);
    assert(pilfflonk_last_error()[0] == '\0' && pilfflonk_last_status() == PILFFLONK_OK);
}

// Keccak-256, not SHA3-256: "" and "abc" are the published vectors (SHA3-256 of "" is a7ffc6f8…);
// the others, bytes i mod 256 of the lengths around the rate (136 bytes), are @noble/hashes'
// keccak_256, an implementation independent of this one.
void testKeccak256() {
    struct Vector {
        uint64_t length;
        const char *hash;
    };
    const Vector vectors[] = {
        {135, "cbdfd9dee5faad3818d6b06f95a219fd290b0e1706f6a82e5a595b9ce9faca62"},
        {136, "7ce759f1ab7f9ce437719970c26b0a66ff11fe3e38e17df89cf5d29c7d7f807e"},
        {137, "ac73d4fae68b8453f764007c1a20ce95994187861f0c3227a3a8e99a73a3b1db"},
        {272, "fdf2ec49e749960d3c8521a0219af8d03e30e2b3bf19bd16150ee0eaf133d66e"},
    };
    // The hashes are written in the order Keccak outputs the bytes, which Bytes32 reverses.
    auto expectHash = [](const uint8_t out[32], const char *hex) {
        Bytes32 expected(hex);
        for (int i = 0; i < 32; ++i) {
            assert(out[i] == expected.bytes[31 - i]);
        }
    };

    uint8_t out[32];
    assert(pilfflonk_keccak256(nullptr, 0, out) == PILFFLONK_OK);
    expectHash(out, "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
    const uint8_t abc[] = {'a', 'b', 'c'};
    assert(pilfflonk_keccak256(abc, sizeof(abc), out) == PILFFLONK_OK);
    expectHash(out, "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");
    uint8_t bytes[272];
    for (size_t i = 0; i < sizeof(bytes); ++i) {
        bytes[i] = static_cast<uint8_t>(i);
    }
    for (const Vector &v : vectors) {
        assert(pilfflonk_keccak256(bytes, v.length, out) == PILFFLONK_OK);
        expectHash(out, v.hash);
    }

    assert(pilfflonk_keccak256(abc, sizeof(abc), nullptr) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(std::strstr(pilfflonk_last_error(), "pilfflonk_keccak256: out is NULL") != nullptr);
    assert(pilfflonk_keccak256(nullptr, 1, out) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(std::strstr(pilfflonk_last_error(), "data is NULL") != nullptr);
    assert(pilfflonk_keccak256(abc, uint64_t(1) << 63, out) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(std::strstr(pilfflonk_last_error(), "exceeds 2^63 - 1") != nullptr);
    assert(pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT);
}

} // namespace

int main() {
    testCheckCanonical();
    testSuccessClearsLastError();
    testKeccak256();
    PilFflonkTest::runTranscriptTests();
    PilFflonkTest::runLdeTests();
    PilFflonkTest::runSrsTests();
    PilFflonkTest::runCommitTests();
    PilFflonkTest::runPolynomialTests();
    PilFflonkTest::runShplonkTests();
    PilFflonkTest::runInfoTests();
    PilFflonkTest::runExpressionsTests();
    PilFflonkTest::runProverTests();
    PilFflonkTest::runGpuTests();
    PilFflonkTest::runExpressionsGpuTests();
    PilFflonkTest::runHintsGpuTests();
    std::printf("pilfflonk_test: all tests passed\n");
    return 0;
}
