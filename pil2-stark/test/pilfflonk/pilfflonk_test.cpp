// Tests for the pilfflonk C API. Run with `make pilfflonk_test`; exits non-zero on the first failure.
#include "pilfflonk_test.hpp"

#include <cstdio>

#include "pilfflonk_api.hpp"

using PilFflonkTest::Bytes32;
using PilFflonkTest::R_HEX;

namespace {

void expectCanonical(const char *hex) {
    const Bytes32 s(hex);
    assert(pilfflonk_fr_check_canonical(s.bytes) == PILFFLONK_OK);
    assert(pilfflonk_last_error()[0] == '\0');
}

void expectNonCanonical(const char *hex) {
    const Bytes32 s(hex);
    assert(pilfflonk_fr_check_canonical(s.bytes) == PILFFLONK_ERR_NON_CANONICAL);
    assert(std::strstr(pilfflonk_last_error(), "pilfflonk_fr_check_canonical") != nullptr);
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

void testNullScalar() {
    assert(pilfflonk_fr_check_canonical(nullptr) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_last_error()[0] != '\0');
}

void testSuccessClearsLastError() {
    expectNonCanonical(R_HEX);
    expectCanonical("0000000000000000000000000000000000000000000000000000000000000000");
}

} // namespace

int main() {
    testCheckCanonical();
    testNullScalar();
    testSuccessClearsLastError();
    PilFflonkTest::runTranscriptTests();
    PilFflonkTest::runLdeTests();
    PilFflonkTest::runSrsTests();
    PilFflonkTest::runCommitTests();
    PilFflonkTest::runShplonkTests();
    PilFflonkTest::runInfoTests();
    std::printf("pilfflonk_test: all tests passed\n");
    return 0;
}
