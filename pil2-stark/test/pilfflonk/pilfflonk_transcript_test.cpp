// Tests for pilfflonk_transcript_*: every challenge must equal the one computed by hand from the
// transcript's encoding (pilfflonk/docs/protocol.md#transcript), and every refused input must come
// back as a status, not an abort.
#include "pilfflonk_test.hpp"

#include <gmp.h>

#include <cstdio>
#include <vector>

#include "alt_bn128.hpp"
#include "keccak_256_transcript.hpp"
#include "keccak_wrapper.hpp"
#include "pilfflonk_api.hpp"

namespace PilFflonkTest {

namespace {

// An affine G1 point as the C API passes it: x‖y, little-endian coordinates.
struct Point {
    Bytes32 x;
    Bytes32 y;

    Point() = default;
    Point(const char *xHex, const char *yHex) : x(xHex), y(yHex) {}
};

const Bytes32 ZERO("0000000000000000000000000000000000000000000000000000000000000000");
const Bytes32 ONE("0000000000000000000000000000000000000000000000000000000000000001");
const Bytes32 R_MINUS_ONE("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000");
const Bytes32 ALL_ONES("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");

// k·G for the generator G = (1, 2), computed independently of ffiasm (affine double-and-add mod q).
const Point P2("030644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd3",
               "15ed738c0e0a7c92e7845f96b2ae9c0a68a6a449e3538fc7ff3ebf7a5a18a2c4");
const Point P3("0769bf9ac56bea3ff40232bcb1b6bd159315d84715b8e679f2d355961915abf0",
               "2ab799bee0489429554fdb7c8d086475319e63b40b9c5b57cdf1ff3dd9fe2261");
const Point P5("17c139df0efee0f766bc0204762b774362e4ded88953a39ce849a8a7fa163fa9",
               "01e0559bacb160664764a357af8a9fe70baa9258e0b959273ffc5718c6d4cc7c");
const Point P7("17072b2ed3bb8d759a5325f477629386cb6fc6ecb801bd76983a6b86abffe078",
               "168ada6cd130dd52017bb54bfa19377aadfe3bf05d18f41b77809f7f60d4af9e");

// Points on the curve with a coordinate below 2^192, which the transcript refuses.
const Point G("0000000000000000000000000000000000000000000000000000000000000001",
              "0000000000000000000000000000000000000000000000000000000000000002");
const Point MINUS_G("0000000000000000000000000000000000000000000000000000000000000001",
                    "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd45");
const Point SHORT_Y("06f01ee8d2eb05e922d444637d6bce667a5424edbe74fae5e74798cf68745a4b",
                    "0000000000000000000000000000000000000000000000000000000000000004");

// The challenges of testPinnedChallenges(), also checked by the Rust wrapper's tests
// (provers/starks-lib-c/src/ffi_pilfflonk.rs): keep both in sync.
const char *const PINNED_HEX[3] = {
    "26ecd6b31f13f24a81bb71f60ef433df1185225122b3b101c736948874b1a41f",
    "11dba4dd6851bff01335d5e5b90efe759da23fce9a86a1c050daefdcd6948c14",
    "1a94293663ade82c5745a10779392a00a28435f34adb91d46084d9e50bee7a69",
};

// keccak256(bytes) mod r, computed without ffiasm: rapidsnark's keccak() -- the hash that
// Keccak256Transcript calls -- and the reduction with GMP.
Bytes32 hashToFr(std::vector<uint8_t> bytes) {
    uint8_t hash[32];
    assert(keccak(bytes.data(), static_cast<int64_t>(bytes.size()), hash, sizeof(hash)) == 32);
    mpz_t value, r;
    mpz_inits(value, r, nullptr);
    mpz_import(value, sizeof(hash), 1, 1, 1, 0, hash);
    assert(mpz_set_str(r, R_HEX, 16) == 0);
    mpz_mod(value, value, r);
    Bytes32 challenge;
    mpz_export(challenge.bytes, nullptr, -1, 1, -1, 0, value);
    mpz_clears(value, r, nullptr);
    return challenge;
}

void appendBigEndian(std::vector<uint8_t> &buffer, const Bytes32 &value) {
    for (int i = 31; i >= 0; --i) {
        buffer.push_back(value.bytes[i]);
    }
}

// The transcript written out by hand: a scalar is 32 big-endian bytes, a point x‖y in 64, and a
// squeeze hashes the buffer to h and restarts it as enc(h).
class Reference {
public:
    void addScalar(const Bytes32 &scalar) { appendBigEndian(buffer, scalar); }

    void addPoint(const Point &point) {
        appendBigEndian(buffer, point.x);
        appendBigEndian(buffer, point.y);
    }

    Bytes32 squeeze() {
        const Bytes32 challenge = hashToFr(buffer);
        buffer.clear();
        addScalar(challenge);
        return challenge;
    }

private:
    std::vector<uint8_t> buffer;
};

std::vector<uint8_t> flatten(const std::vector<Bytes32> &scalars) {
    std::vector<uint8_t> bytes;
    for (const Bytes32 &scalar : scalars) {
        bytes.insert(bytes.end(), scalar.bytes, scalar.bytes + sizeof(scalar.bytes));
    }
    return bytes;
}

std::vector<uint8_t> flatten(const std::vector<Point> &points) {
    std::vector<uint8_t> bytes;
    for (const Point &point : points) {
        bytes.insert(bytes.end(), point.x.bytes, point.x.bytes + sizeof(point.x.bytes));
        bytes.insert(bytes.end(), point.y.bytes, point.y.bytes + sizeof(point.y.bytes));
    }
    return bytes;
}

bool lastErrorMentions(const char *text) {
    return std::strstr(pilfflonk_last_error(), text) != nullptr;
}

// A transcript handle of the C API.
class Api {
public:
    Api() : handle(pilfflonk_transcript_new()) {
        assert(handle != nullptr);
        assert(pilfflonk_last_error()[0] == '\0');
    }
    ~Api() { pilfflonk_transcript_free(handle); }
    Api(const Api &) = delete;
    Api &operator=(const Api &) = delete;

    int absorb(const std::vector<Bytes32> &scalars) {
        const std::vector<uint8_t> bytes = flatten(scalars);
        return pilfflonk_transcript_absorb(handle, bytes.data(), scalars.size(), PILFFLONK_TRANSCRIPT_FR);
    }

    int absorb(const std::vector<Point> &points) {
        const std::vector<uint8_t> bytes = flatten(points);
        return pilfflonk_transcript_absorb(handle, bytes.data(), points.size(), PILFFLONK_TRANSCRIPT_G1);
    }

    Bytes32 squeeze() {
        Bytes32 challenge;
        assert(pilfflonk_transcript_squeeze(handle, challenge.bytes) == PILFFLONK_OK);
        assert(pilfflonk_last_error()[0] == '\0');
        assert(pilfflonk_fr_check_canonical(challenge.bytes) == PILFFLONK_OK);
        return challenge;
    }

    void *const handle;
};

// Makes the same calls on the C API and on the reference; every squeeze must agree.
class Checked {
public:
    Checked &scalars(const std::vector<Bytes32> &values) {
        assert(api.absorb(values) == PILFFLONK_OK);
        assert(pilfflonk_last_error()[0] == '\0');
        for (const Bytes32 &value : values) {
            reference.addScalar(value);
        }
        return *this;
    }

    Checked &points(const std::vector<Point> &values) {
        assert(api.absorb(values) == PILFFLONK_OK);
        assert(pilfflonk_last_error()[0] == '\0');
        for (const Point &value : values) {
            reference.addPoint(value);
        }
        return *this;
    }

    Bytes32 squeeze() {
        const Bytes32 challenge = api.squeeze();
        assert(challenge == reference.squeeze());
        return challenge;
    }

    Api api;

private:
    Reference reference;
};

// Deterministic scalars below 2^253 < r, from a keccak chain.
std::vector<Bytes32> someScalars(size_t n) {
    std::vector<Bytes32> scalars(n);
    uint8_t state[32] = {};
    for (Bytes32 &scalar : scalars) {
        assert(keccak(state, sizeof(state), state, sizeof(state)) == 32);
        std::memcpy(scalar.bytes, state, sizeof(state));
        scalar.bytes[31] &= 0x1f;
    }
    return scalars;
}

// k·G for each k, through ffiasm: only to have many inputs, which the C API then checks.
std::vector<Point> somePoints(const std::vector<Bytes32> &multipliers) {
    AltBn128::Engine &E = AltBn128::Engine::engine;
    std::vector<Point> points(multipliers.size());
    for (size_t i = 0; i < multipliers.size(); ++i) {
        AltBn128::Engine::G1Point product;
        Bytes32 k = multipliers[i];
        E.g1.mulByScalar(product, E.g1.oneAffine(), k.bytes, sizeof(k.bytes));
        AltBn128::Engine::G1PointAffine affine;
        E.g1.copy(affine, product);
        E.f1.toRprLE(affine.x, points[i].x.bytes, sizeof(points[i].x.bytes));
        E.f1.toRprLE(affine.y, points[i].y.bytes, sizeof(points[i].y.bytes));
    }
    return points;
}

// The hash is Keccak-256 (Ethereum's), not SHA3-256: both are 32 bytes, only the padding differs.
void testKeccakIsKeccak256() {
    uint8_t hash[32];
    std::vector<uint8_t> abc = {'a', 'b', 'c'};
    assert(keccak(abc.data(), static_cast<int64_t>(abc.size()), hash, sizeof(hash)) == 32);
    Bytes32 expected("4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");
    for (int i = 0; i < 32; ++i) {
        assert(hash[i] == expected.bytes[31 - i]);
    }
}

void testMixedSequences() {
    // FflonkProver's round 2: a point, scalars, a point; beta, then gamma from beta alone.
    {
        Checked t;
        t.points({P2}).scalars({ONE, R_MINUS_ONE, ZERO}).points({P3});
        const Bytes32 beta = t.squeeze();
        const Bytes32 gamma = t.squeeze();
        assert(beta != gamma);
        // An earlier challenge absorbed back, points after scalars, then several squeezes in a row.
        t.scalars({beta}).points({P5, P7}).scalars({gamma});
        const Bytes32 a = t.squeeze();
        const Bytes32 b = t.squeeze();
        const Bytes32 c = t.squeeze();
        assert(a != b && b != c && a != c);
    }
    // Points first, and one element at a time.
    {
        Checked t;
        t.points({P7}).scalars({ZERO}).points({P2}).scalars({ONE});
        t.squeeze();
        t.points({P3});
        t.squeeze();
    }
    // Many elements per call: the buffer Keccak256Transcript sizes for them must be right.
    {
        Checked t;
        const std::vector<Bytes32> scalars = someScalars(300);
        const std::vector<Point> points = somePoints(someScalars(40));
        t.scalars(scalars).points(points).scalars({R_MINUS_ONE});
        t.squeeze();
        t.points(points).scalars(scalars);
        t.squeeze();
    }
}

// The sequence whose challenges PINNED_HEX records (also computed with an independent Keccak-256
// outside this repository's code when it was pinned).
void testPinnedChallenges() {
    Checked t;
    t.scalars({ONE, R_MINUS_ONE}).points({P2, P3});
    assert(t.squeeze() == Bytes32(PINNED_HEX[0]));
    t.points({P5});
    assert(t.squeeze() == Bytes32(PINNED_HEX[1]));
    assert(t.squeeze() == Bytes32(PINNED_HEX[2]));
}

// One call with n elements hashes exactly as n calls with one, and n = 0 changes nothing.
void testBatchingDoesNotChangeTheChallenge() {
    Api batched;
    assert(batched.absorb(std::vector<Bytes32>{ONE, ZERO, R_MINUS_ONE}) == PILFFLONK_OK);
    assert(batched.absorb(std::vector<Point>{P2, P3}) == PILFFLONK_OK);

    Api single;
    for (const Bytes32 &scalar : {ONE, ZERO, R_MINUS_ONE}) {
        assert(single.absorb(std::vector<Bytes32>{scalar}) == PILFFLONK_OK);
        assert(pilfflonk_transcript_absorb(single.handle, nullptr, 0, PILFFLONK_TRANSCRIPT_G1) == PILFFLONK_OK);
    }
    for (const Point &point : {P2, P3}) {
        assert(single.absorb(std::vector<Point>{point}) == PILFFLONK_OK);
        assert(pilfflonk_transcript_absorb(single.handle, nullptr, 0, PILFFLONK_TRANSCRIPT_FR) == PILFFLONK_OK);
    }
    assert(batched.squeeze() == single.squeeze());
}

// After each refused call the transcript must be as if the call had not been made.
void testRefusedArguments() {
    Checked t;
    t.scalars({ONE});
    uint8_t bytes[64] = {};
    Bytes32 out;

    assert(pilfflonk_transcript_absorb(nullptr, bytes, 1, PILFFLONK_TRANSCRIPT_FR) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(lastErrorMentions("pilfflonk_transcript_absorb") && lastErrorMentions("transcript is NULL"));
    assert(pilfflonk_transcript_absorb(t.api.handle, nullptr, 1, PILFFLONK_TRANSCRIPT_G1) ==
           PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(lastErrorMentions("data is NULL"));
    for (uint32_t kind : {2u, 0xffffffffu}) {
        assert(pilfflonk_transcript_absorb(t.api.handle, bytes, 1, kind) == PILFFLONK_ERR_INVALID_ARGUMENT);
        assert(lastErrorMentions("kind"));
        assert(pilfflonk_transcript_absorb(t.api.handle, bytes, 0, kind) == PILFFLONK_ERR_INVALID_ARGUMENT);
    }
    // More than the 2^31 - 1 bytes Keccak256Transcript can hash: refused before `bytes` is read,
    // so the short buffer is never overrun.
    for (uint64_t n : {uint64_t(1) << 26, uint64_t(UINT64_MAX)}) {
        assert(pilfflonk_transcript_absorb(t.api.handle, bytes, n, PILFFLONK_TRANSCRIPT_FR) ==
               PILFFLONK_ERR_INVALID_ARGUMENT);
        assert(lastErrorMentions("exceed"));
    }
    for (uint64_t n : {uint64_t(11184811), uint64_t(UINT64_MAX)}) { // 11184811 * 192 > 2^31 - 1
        assert(pilfflonk_transcript_absorb(t.api.handle, bytes, n, PILFFLONK_TRANSCRIPT_G1) ==
               PILFFLONK_ERR_INVALID_ARGUMENT);
        assert(lastErrorMentions("exceed"));
    }

    assert(pilfflonk_transcript_squeeze(nullptr, out.bytes) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(lastErrorMentions("pilfflonk_transcript_squeeze") && lastErrorMentions("transcript is NULL"));
    assert(pilfflonk_transcript_squeeze(t.api.handle, nullptr) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(lastErrorMentions("out is NULL"));

    t.squeeze();

    // Nothing to hash yet: snarkjs's Keccak256Transcript throws, and getChallenge() would declare
    // a zero-length array.
    Api empty;
    assert(pilfflonk_transcript_squeeze(empty.handle, out.bytes) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(lastErrorMentions("nothing has been absorbed"));
    assert(pilfflonk_transcript_absorb(empty.handle, nullptr, 0, PILFFLONK_TRANSCRIPT_FR) == PILFFLONK_OK);
    assert(pilfflonk_transcript_squeeze(empty.handle, out.bytes) == PILFFLONK_ERR_INVALID_ARGUMENT);

    pilfflonk_transcript_free(nullptr);
    assert(pilfflonk_last_error()[0] == '\0');
}

void expectRefused(Checked &t, const std::vector<Bytes32> &scalars, int status, const char *message) {
    assert(t.api.absorb(scalars) == status);
    assert(lastErrorMentions("pilfflonk_transcript_absorb") && lastErrorMentions(message));
}

void expectRefused(Checked &t, const std::vector<Point> &points, int status, const char *message) {
    assert(t.api.absorb(points) == status);
    assert(lastErrorMentions("pilfflonk_transcript_absorb") && lastErrorMentions(message));
}

void testRefusedScalars() {
    Checked t;
    t.points({P2});
    expectRefused(t, {Bytes32(R_HEX)}, PILFFLONK_ERR_NON_CANONICAL, "element 0 is not canonical");
    expectRefused(t, {ALL_ONES}, PILFFLONK_ERR_NON_CANONICAL, "element 0 is not canonical");
    // All or nothing: the valid scalars before the bad one are not absorbed either.
    expectRefused(t, {ONE, R_MINUS_ONE, Bytes32(R_HEX)}, PILFFLONK_ERR_NON_CANONICAL, "element 2 is not canonical");
    t.scalars({ONE});
    t.squeeze();
}

void testRefusedPoints() {
    Checked t;
    t.scalars({ONE});
    Point nonCanonicalX = P2; // x + q: on the curve once reduced, so it must not be reduced silently
    nonCanonicalX.x = Bytes32("336a935a0f44ba2c53d54a11e9996de370f9813a7ef8e7360fe294d84604cd1a");
    Point xIsQ = P2;
    xIsQ.x = Bytes32(Q_HEX);
    Point yAllOnes = P2;
    yAllOnes.y = ALL_ONES;
    for (const Point &point : {nonCanonicalX, xIsQ, yAllOnes}) {
        expectRefused(t, {point}, PILFFLONK_ERR_NON_CANONICAL, "element 0 is not canonical");
    }

    Point offByOne = P2;
    offByOne.y = Bytes32("15ed738c0e0a7c92e7845f96b2ae9c0a68a6a449e3538fc7ff3ebf7a5a18a2c5");
    const Point offCurve("0000000000000000000000000000000000000000000000000000000000000001",
                         "0000000000000000000000000000000000000000000000000000000000000003");
    for (const Point &point : {offByOne, offCurve}) {
        expectRefused(t, {point}, PILFFLONK_ERR_INVALID_POINT, "element 0 is not on the curve");
    }

    const Point infinity; // (0, 0)
    expectRefused(t, {infinity}, PILFFLONK_ERR_INVALID_POINT, "element 0 is (0, 0), the point at infinity");

    for (const Point &point : {G, MINUS_G, SHORT_Y}) {
        expectRefused(t, {point}, PILFFLONK_ERR_INVALID_POINT, "element 0 has a coordinate below 2^192");
    }

    // All or nothing, as for scalars.
    expectRefused(t, {P3, P5, offCurve}, PILFFLONK_ERR_INVALID_POINT, "element 2 is not on the curve");
    t.points({P7});
    t.squeeze();
}

Bytes32 toBytes(AltBn128::Engine::FrElement element) {
    Bytes32 bytes;
    AltBn128::Engine::engine.fr.toRprLE(element, bytes.bytes, sizeof(bytes.bytes));
    return bytes;
}

// Why the C API refuses the two kinds of points above: what Keccak256Transcript itself hashes for
// them. If this test starts failing, the class or ffiasm changed and the refusal may be revisited.
void testKeccak256TranscriptQuirks() {
    AltBn128::Engine &E = AltBn128::Engine::engine;

    // G = (1, 2): RawFq::toRprBE writes each coordinate as its one significant 64-bit word,
    // left-aligned, so the buffer holds 1·2^192 and 2·2^192 instead of 1 and 2.
    {
        Keccak256Transcript<AltBn128::Engine> transcript(E);
        AltBn128::Engine::G1Point g;
        E.g1.copy(g, E.g1.oneAffine());
        transcript.addPolCommitment(g);
        const Bytes32 challenge = toBytes(transcript.getChallenge());

        std::vector<uint8_t> leftAligned(64, 0);
        leftAligned[7] = 1;
        leftAligned[32 + 7] = 2;
        assert(challenge == hashToFr(leftAligned));

        Reference a4;
        a4.addPoint(G);
        assert(challenge != a4.squeeze());
    }

    // The point at infinity clears the first 64 bytes of the buffer and appends nothing: a scalar
    // followed by it hashes as one zero scalar.
    {
        Keccak256Transcript<AltBn128::Engine> transcript(E);
        AltBn128::Engine::FrElement five;
        E.fr.fromUI(five, 5);
        transcript.addScalar(five);
        transcript.addPolCommitment(E.g1.zero());
        const Bytes32 challenge = toBytes(transcript.getChallenge());

        Reference zero;
        zero.addScalar(ZERO);
        assert(challenge == zero.squeeze());
    }
}

} // namespace

void runTranscriptTests() {
    testKeccakIsKeccak256();
    testMixedSequences();
    testPinnedChallenges();
    testBatchingDoesNotChangeTheChallenge();
    testRefusedArguments();
    testRefusedScalars();
    testRefusedPoints();
    testKeccak256TranscriptQuirks();
}

} // namespace PilFflonkTest
