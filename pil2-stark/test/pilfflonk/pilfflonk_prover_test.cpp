// Tests for the prover (plan M18): PilFflonk::BlindingRng, AirKey and ProvingKey, Instance, Opening,
// and the C API over them.
//
// The AIR is the Fibonacci of M13 as setup-pilfflonk writes it, from the fixtures of M17
// (setup/pilfflonk/tests/fixtures/bytecode/fibonacci/: its pilfflonkinfo, .bin and .const, M13's
// witness, the qVerifier as a bytecode, and the oracle's (M14) evaluations and Q at a point). Each
// test writes a provingKey/ around them, with a globalInfo of that AIR and an SRS of the test ptau,
// whose τ it knows: so it can check every commitment against [p(τ)]₁ and the whole SHPLONK opening
// against the identity of A.5, F − E − J + y·W' = τ·W', as the verifier's pairing would.
//
// - With the blinding at zero (a BlindingSource of the tests: the C API has none), the prover's
//   evaluations at the oracle's ξ and its Q(ξ) are the oracle's (M14), exactly.
// - With it on (seeded, D6): the same seed gives the same proof, another seed other commitments;
//   each committed polynomial still is its column on H, has its |O| + 1 more coefficients, and Q(ξ)
//   is the verifier's, the qVerifier's at the blinded evaluations.
// - A mutated witness makes commitQ fail with UnsatisfiedError (PILFFLONK_ERR_UNSATISFIED).
#include "pilfflonk_test.hpp"
#include "pilfflonk_test_ptau.hpp"

#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pilfflonk_api.hpp"
#include "pilfflonk_error.hpp"
#include "pilfflonk_expressions.hpp"
#include "pilfflonk_expressions_bin.hpp"
#include "pilfflonk_prover.hpp"
#include "pilfflonk_proving_key.hpp"
#include "pilfflonk_rng.hpp"
#include "pilfflonk_srs.hpp"
#include "pilfflonk_transcript.hpp"

namespace PilFflonkTest {

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using PilFflonk::AirKey;
using PilFflonk::BlindingRng;
using PilFflonk::BlindingSource;
using PilFflonk::ColumnRead;
using PilFflonk::Expressions;
using PilFflonk::ExpressionsBin;
using PilFflonk::FormatError;
using PilFflonk::FrElement;
using PilFflonk::G1Point;
using PilFflonk::Instance;
using PilFflonk::IoError;
using PilFflonk::LayoutEntry;
using PilFflonk::Opening;
using PilFflonk::PilfflonkInfo;
using PilFflonk::PointValues;
using PilFflonk::Poly;
using PilFflonk::ProvingKey;
using PilFflonk::Srs;
using PilFflonk::Transcript;
using PilFflonk::UnsatisfiedError;
using Engine = AltBn128::Engine;

Engine &E = Engine::engine;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

std::string exeDir() {
    char exe[PATH_MAX];
    const ssize_t length = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    assert(length > 0);
    exe[length] = '\0';
    std::string dir(exe);
    return dir.substr(0, dir.rfind('/'));
}

std::string fixture(const std::string &name) {
    const char *root = std::getenv("PILFFLONK_REPO_ROOT");
    const std::string repo = root != nullptr ? std::string(root) : exeDir() + "/../..";
    return repo + "/setup/pilfflonk/tests/fixtures/bytecode/fibonacci/" + name;
}

std::vector<uint8_t> readBytes(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    assert(file);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

void writeBytes(const std::string &path, const std::vector<uint8_t> &bytes) {
    std::ofstream file(path, std::ios::binary);
    assert(file);
    file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    assert(file);
}

std::vector<uint8_t> text(const std::string &s) { return std::vector<uint8_t>(s.begin(), s.end()); }

bool contains(const std::string &s, const std::string &part) { return s.find(part) != std::string::npos; }

template <typename Exception, typename Call>
std::string thrown(Call call) {
    try {
        call();
    } catch (const Exception &e) {
        return e.what();
    }
    assert(!"expected an exception");
    return "";
}

FrElement fr(const json &decimal) {
    FrElement e;
    E.fr.fromString(e, decimal.get<std::string>(), 10);
    return e;
}

std::vector<FrElement> frs(const json &values) {
    std::vector<FrElement> out;
    for (const json &v : values) out.push_back(fr(v));
    return out;
}

bool eq(const FrElement &a, const FrElement &b) { return E.fr.eq(a, b); }

FrElement power(const FrElement &base, uint64_t exponent) {
    uint8_t bytes[sizeof(exponent)];
    for (size_t i = 0; i < sizeof(exponent); ++i) bytes[i] = static_cast<uint8_t>(exponent >> (8 * i));
    FrElement r;
    E.fr.exp(r, base, bytes, sizeof(bytes));
    return r;
}

FrElement inverse(const FrElement &a) {
    FrElement r;
    E.fr.inv(r, a);
    return r;
}

G1Point times(G1Point p, const FrElement &x) {
    FrElement canonical;
    E.fr.fromMontgomery(canonical, x);
    G1Point r;
    E.g1.mulByScalar(r, p, reinterpret_cast<uint8_t *>(canonical.v), sizeof(canonical.v));
    return r;
}

G1Point g1Times(const FrElement &x) {
    G1Point g;
    E.g1.copy(g, E.g1.oneAffine());
    return times(g, x);
}

bool samePoint(G1Point a, G1Point b) { return E.g1.eq(a, b); }

G1Point add(G1Point a, G1Point b) {
    G1Point r;
    E.g1.add(r, a, b);
    return r;
}

G1Point sub(G1Point a, G1Point b) {
    G1Point r;
    E.g1.sub(r, a, b);
    return r;
}

// The blinding factors at zero: the polynomials are then the columns' interpolants, as the oracle
// computes them. Only a test can do this; the C API always blinds (D6).
class ZeroBlinding final : public BlindingSource {
public:
    void fill(FrElement *out, uint64_t n) override {
        for (uint64_t i = 0; i < n; ++i) out[i] = E.fr.zero();
    }
};

// ---------------------------------------------------------------------------------------------
// A provingKey/ of the Fibonacci
// ---------------------------------------------------------------------------------------------

constexpr const char *NAME = "fib";
constexpr const char *AIRGROUP = "Fibonacci";
constexpr const char *AIR = "Fibonacci";
constexpr uint64_t N_G1 = 512;

std::string globalInfoJson(const std::string &backend = "pilfflonk", uint64_t numRows = 256) {
    return "{\"name\": \"" + std::string(NAME) + "\", \"airs\": [[{\"name\": \"" + AIR +
           "\", \"num_rows\": " + std::to_string(numRows) + "}]], \"air_groups\": [\"" + AIRGROUP +
           "\"], \"aggTypes\": [[]], \"backend\": \"" + backend +
           "\", \"formatVersion\": 1, \"field\": \"bn254\", \"nPublics\": 3, \"numChallenges\": [0], "
           "\"numProofValues\": [], \"proofValuesMap\": [], \"publicsMap\": []}";
}

// The SRS of the test ptau, with its first nG1 powers.
std::vector<uint8_t> srsBytes(uint64_t nG1) {
    TestDir dir;
    const std::string ptau = dir.file("prover.ptau");
    const std::string srs = dir.file("prover.srs.bin");
    writeTestPtau(ptau, nG1);
    Srs::fromPtau(ptau, nG1).save(srs);
    return readBytes(srs);
}

const std::vector<uint8_t> &defaultSrs() {
    static const std::vector<uint8_t> bytes = srsBytes(N_G1);
    return bytes;
}

// The files of a provingKey/, which a test may change before writing them.
struct KeyFiles {
    std::vector<uint8_t> globalInfo = text(globalInfoJson());
    std::vector<uint8_t> srs = defaultSrs();
    std::vector<uint8_t> info = readBytes(fixture("Fibonacci.pilfflonkinfo.json"));
    std::vector<uint8_t> bin = readBytes(fixture("Fibonacci.bin"));
    std::vector<uint8_t> constants = readBytes(fixture("Fibonacci.const"));
};

// A provingKey/ in a fresh directory next to the test binary, removed with it.
class KeyDir {
public:
    explicit KeyDir(const KeyFiles &files = KeyFiles()) {
        std::string pattern = exeDir() + "/pilfflonk_prover.XXXXXX";
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        assert(mkdtemp(buffer.data()) != nullptr);
        root = buffer.data();
        const std::string backend = root + "/" + NAME + "/pilfflonk";
        const std::string air = root + "/" + NAME + "/" + AIRGROUP + "/airs/" + AIR + "/air";
        fs::create_directories(backend);
        fs::create_directories(air);
        writeBytes(root + "/pilout.globalInfo.json", files.globalInfo);
        writeBytes(backend + "/pilfflonk.srs.bin", files.srs);
        writeBytes(air + "/" + AIR + ".pilfflonkinfo.json", files.info);
        writeBytes(air + "/" + AIR + ".bin", files.bin);
        writeBytes(air + "/" + AIR + ".const", files.constants);
    }
    ~KeyDir() { fs::remove_all(root); }
    KeyDir(const KeyDir &) = delete;
    KeyDir &operator=(const KeyDir &) = delete;

    const std::string &path() const { return root; }
    std::string airFile(const std::string &extension) const {
        return root + "/" + NAME + "/" + AIRGROUP + "/airs/" + AIR + "/air/" + AIR + "." + extension;
    }

private:
    std::string root;
};

// What every test of the Fibonacci starts from.
struct Fibonacci {
    KeyDir dir;
    std::unique_ptr<ProvingKey> pk = ProvingKey::load(dir.path());
    json oracle = json::parse(readBytes(fixture("Fibonacci.oracle.json")));
    std::vector<uint8_t> witness = readBytes(fixture("Fibonacci.witness.bin"));
    std::vector<FrElement> publics = frs(oracle["publics"]);

    const AirKey &air() const { return pk->air(0, 0); }

    std::unique_ptr<Instance> instance(std::unique_ptr<BlindingSource> blinding,
                                       const std::vector<uint8_t> &trace) const {
        return std::make_unique<Instance>(*pk, 0, 0, trace.data(), trace.size(), std::vector<FrElement>{}, publics,
                                          std::vector<FrElement>{}, std::move(blinding));
    }
    std::unique_ptr<Instance> instance(std::unique_ptr<BlindingSource> blinding) const {
        return instance(std::move(blinding), witness);
    }
};

// A proof of the Fibonacci's instance as the orchestrator drives it (A.4 in miniature: a scalar in
// place of the digest and the publics), and everything the checks below need.
struct Proved {
    std::unique_ptr<Instance> instance;
    std::unique_ptr<Opening> opening;
    std::vector<G1Point> commitments; // the non-fixed f, in the order of the layout
    FrElement stdVc, xiSeed;
    Opening::Proof proof;
};

Proved prove(const Fibonacci &fib, std::unique_ptr<BlindingSource> blinding) {
    Proved p;
    p.instance = fib.instance(std::move(blinding));
    Transcript t;
    t.absorb(std::vector<FrElement>{E.fr.one()});
    t.absorb(fib.publics);
    std::vector<G1Point> stage1 = p.instance->commitStage(1, {});
    t.absorb(stage1);
    p.stdVc = t.squeeze();
    std::vector<G1Point> q = p.instance->commitQ({p.stdVc});
    t.absorb(q);
    p.xiSeed = t.squeeze();
    p.commitments = stage1;
    p.commitments.insert(p.commitments.end(), q.begin(), q.end());
    p.opening = std::make_unique<Opening>(std::vector<const Instance *>{p.instance.get()}, p.xiSeed);
    t.absorb(p.opening->evaluations());
    p.proof = p.opening->open(t);
    return p;
}

// Where evaluation e of the proof (A.4 step 4 order) is, as (f, offset): the evMap's const entries
// in order, then its cm entries.
std::vector<std::pair<uint64_t, int64_t>> evaluationPlaces(const AirKey &air) {
    std::vector<std::pair<uint64_t, int64_t>> places;
    for (PilFflonk::PolType type : {PilFflonk::PolType::Const, PilFflonk::PolType::Cm}) {
        for (const PilFflonk::EvMapEntry &e : air.info().evMap) {
            if (e.type == type) {
                const uint64_t f = (type == PilFflonk::PolType::Const ? air.constPosition(e.id) : air.cmPosition(e.id)).f;
                places.push_back({f, e.prime});
            }
        }
    }
    return places;
}

// The verifier's Q(ξ): the qVerifier (M17's fixture) at ξ from the proof's evaluations.
FrElement qVerifierAt(const Fibonacci &fib, const std::vector<FrElement> &evaluations, const FrElement &stdVc,
                      const FrElement &xiSeed, const FrElement &xi) {
    const ExpressionsBin qVerifier = ExpressionsBin::load(fixture("Fibonacci.qverifier.bin"));
    const Expressions expressions(qVerifier, fib.air().info());
    PointValues point;
    point.zerofiers = PilFflonk::zerofiersAt(fib.air().info().nBits, fib.air().info().boundaries, xi);
    // The evMap order: the Fibonacci's has its const entries first, as the proof's order.
    point.evals = evaluations;
    point.publics = fib.publics;
    point.challenges = {stdVc, xiSeed};
    return expressions.evaluateExpressionAt(fib.air().info().cExpId, point);
}

// The verifier's SHPLONK check of A.5 with τ known (every f has k = 1, as --no-packing's layout):
// F − E − J + y·W' = τ·W', with the challenges α and y the opening squeezed, the fixed commitments
// computed here, and the value of Q's f at ξ the verifier's.
bool shplonkIdentityHolds(const Fibonacci &fib, const Proved &p) {
    const AirKey &air = fib.air();
    const std::vector<LayoutEntry> &layout = air.info().layout;
    const uint64_t N = air.n();
    const FrElement xi = p.xiSeed; // powerW = 1
    const FrElement w = PilFflonk::rootOfUnity(air.info().nBits);
    const FrElement alpha = p.proof.shplonk.alpha, y = p.proof.shplonk.y;

    // f_i's commitment, roots and values at them.
    std::vector<G1Point> commitments;
    for (uint64_t f = 0; f < air.nFixedF(); ++f) {
        assert(layout[f].k == 1);
        commitments.push_back(fib.pk->srs().commit(air.fixedPolynomial(layout[f].pols[0].id)->coef, N));
    }
    commitments.insert(commitments.end(), p.commitments.begin(), p.commitments.end());
    std::vector<std::vector<FrElement>> roots(layout.size()), values(layout.size());
    for (uint64_t f = 0; f < layout.size(); ++f) {
        for (int64_t s : layout[f].offsets) {
            roots[f].push_back(E.fr.mul(xi, power(w, static_cast<uint64_t>(s))));
            values[f].push_back(E.fr.zero());
        }
    }
    const std::vector<std::pair<uint64_t, int64_t>> places = evaluationPlaces(air);
    const std::vector<FrElement> &evaluations = p.opening->evaluations();
    for (uint64_t e = 0; e < places.size(); ++e) {
        const std::vector<int64_t> &offsets = layout[places[e].first].offsets;
        const uint64_t m = std::find(offsets.begin(), offsets.end(), places[e].second) - offsets.begin();
        values[places[e].first][m] = evaluations[e];
    }
    values[air.qF()][0] = qVerifierAt(fib, evaluations, p.stdVc, p.xiSeed, xi);

    // Z_{T_i}(y), r_i(y) by Lagrange, and q_i.
    std::vector<FrElement> z(layout.size()), r(layout.size()), q(layout.size());
    for (uint64_t f = 0; f < layout.size(); ++f) {
        z[f] = E.fr.one();
        for (const FrElement &x : roots[f]) E.fr.mul(z[f], z[f], E.fr.sub(y, x));
        r[f] = E.fr.zero();
        for (uint64_t m = 0; m < roots[f].size(); ++m) {
            FrElement basis = E.fr.one();
            for (uint64_t l = 0; l < roots[f].size(); ++l) {
                if (l == m) continue;
                E.fr.mul(basis, basis, E.fr.mul(E.fr.sub(y, roots[f][l]), inverse(E.fr.sub(roots[f][m], roots[f][l]))));
            }
            E.fr.add(r[f], r[f], E.fr.mul(values[f][m], basis));
        }
    }
    FrElement alphaPower = E.fr.one();
    q[0] = z[0];
    for (uint64_t f = 1; f < layout.size(); ++f) {
        E.fr.mul(alphaPower, alphaPower, alpha);
        q[f] = E.fr.mul(alphaPower, E.fr.mul(z[0], inverse(z[f])));
    }

    G1Point F = commitments[0];
    FrElement e = r[0];
    for (uint64_t f = 1; f < layout.size(); ++f) {
        F = add(F, times(commitments[f], q[f]));
        E.fr.add(e, e, E.fr.mul(q[f], r[f]));
    }
    const G1Point lhs =
        add(sub(sub(F, g1Times(e)), times(p.proof.shplonk.w, q[0])), times(p.proof.shplonk.wp, y));
    return samePoint(lhs, times(p.proof.shplonk.wp, testTau()));
}

// ---------------------------------------------------------------------------------------------
// BlindingRng
// ---------------------------------------------------------------------------------------------

std::vector<FrElement> draw(BlindingSource &rng, uint64_t n) {
    std::vector<FrElement> out(n);
    rng.fill(out.data(), n);
    return out;
}

bool sameElements(const std::vector<FrElement> &a, const std::vector<FrElement> &b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), eq);
}

void testBlindingRng() {
    uint8_t seed[BlindingRng::SEED_BYTES];
    for (size_t i = 0; i < sizeof(seed); ++i) seed[i] = static_cast<uint8_t>(i + 1);
    BlindingRng a(seed), b(seed);
    assert(a.seeded() && !BlindingRng().seeded());
    // 400 elements: about 530 candidates of 32 bytes, over five blocks of 4096.
    const std::vector<FrElement> first = draw(a, 400);
    assert(sameElements(first, draw(b, 400)));
    // In pieces, the same stream.
    BlindingRng c(seed);
    std::vector<FrElement> pieces = draw(c, 1);
    for (uint64_t n : {3, 0, 96, 300}) {
        const std::vector<FrElement> more = draw(c, n);
        pieces.insert(pieces.end(), more.begin(), more.end());
    }
    assert(sameElements(first, pieces));

    // Canonical and spread: every element is a scalar below r, and no two are the same.
    for (uint64_t i = 0; i < first.size(); ++i) {
        uint8_t bytes[32];
        PilFflonk::encodeFr(first[i], bytes);
        FrElement back;
        assert(PilFflonk::decodeFr(bytes, back) == PilFflonk::AbsorbError::None && eq(back, first[i]));
        for (uint64_t j = 0; j < i; ++j) assert(!eq(first[i], first[j]));
    }

    // Another seed, another stream; and the OS's randomness, another each time.
    seed[31] ^= 1;
    BlindingRng other(seed);
    assert(!eq(draw(other, 1)[0], first[0]));
    BlindingRng r1, r2;
    assert(!eq(draw(r1, 1)[0], draw(r2, 1)[0]));
}

// ---------------------------------------------------------------------------------------------
// The proving key
// ---------------------------------------------------------------------------------------------

void testLoadsTheFibonaccisKey() {
    const Fibonacci fib;
    const AirKey &air = fib.air();
    const PilFflonk::AirDegrees &d = air.degrees();
    // A.1: N = 256, |O|max = 2 (l1 and l2 at {0, 1}), qDeg = 1: 256 + 2·2 + 1 = 261 coefficients,
    // 2^9 points; M16's nBitsExt.
    assert(d.n == 256 && d.maxOpenings == 2 && d.qCoefficients == 261 && d.nBitsExt == 9);
    assert(air.lde().domainSize() == 256 && air.lde().extendedSize() == 512);
    assert(air.nFixedF() == 2 && air.qF() == 5);
    assert((air.witnessColumns() == std::vector<uint64_t>{0, 1}));
    // The blinding of A.3, |O| + 1 per committed column: M16's degree − N.
    const uint64_t blind[] = {0, 0, 3, 3, 2, 0};
    for (uint64_t f = 0; f < 6; ++f) {
        assert(air.blindLength(f) == blind[f]);
        if (f >= 2 && f < 5) assert(air.info().layout[f].degree - air.n() == blind[f]);
    }
    assert(air.constPosition(1).f == 1 && air.cmPosition(2).f == 4 && air.cmPosition(3).f == 5);
    // Q reads the fixed columns and the three of stage 1.
    const std::vector<ColumnRead> expected = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {1, 2}};
    assert(air.qReads() == expected);
    // The fixed columns' interpolants give back the .const on H.
    const FrElement w = PilFflonk::rootOfUnity(8);
    for (uint64_t c = 0; c < 2; ++c) {
        for (uint64_t i : {0, 1, 100, 255}) {
            assert(eq(air.fixedPolynomial(c)->fastEvaluate(power(w, i)), air.fixedEvaluations(c)[i]));
        }
    }
    assert(contains(thrown<std::invalid_argument>([&] { fib.pk->air(0, 1); }), "no air 1 in airgroup 0"));
    assert(contains(thrown<std::invalid_argument>([&] { fib.pk->air(1, 0); }), "no air 0 in airgroup 1"));

    // Through the C API.
    void *ctx = pilfflonk_ctx_new(fib.dir.path().c_str());
    assert(ctx != nullptr);
    uint64_t nBitsExt = 0;
    assert(pilfflonk_ctx_n_bits_ext(ctx, 0, 0, &nBitsExt) == PILFFLONK_OK && nBitsExt == 9);
    assert(pilfflonk_ctx_n_bits_ext(ctx, 0, 1, &nBitsExt) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_ctx_n_bits_ext(ctx, 0, 0, nullptr) == PILFFLONK_ERR_INVALID_ARGUMENT);

    // The commitments of the fixed f, unpacked here (k = 1): [p(τ)]₁ of each fixed column's
    // interpolant, what the setup writes in the vkey, from AirKey and through the C API.
    const std::vector<G1Point> fixed = air.fixedCommitments(fib.pk->srs());
    assert(fixed.size() == 2);
    std::vector<uint8_t> fixedBytes(2 * PilFflonk::G1_BYTES);
    assert(pilfflonk_ctx_fixed_commitments(ctx, 0, 0, fixedBytes.data(), 2) == PILFFLONK_OK);
    for (uint64_t c = 0; c < 2; ++c) {
        const G1Point expected = g1Times(air.fixedPolynomial(c)->evaluate(testTau()));
        assert(samePoint(fixed[c], expected));
        uint8_t encoded[PilFflonk::G1_BYTES];
        PilFflonk::encodeG1(expected, encoded);
        assert(std::equal(encoded, encoded + PilFflonk::G1_BYTES, fixedBytes.begin() + c * PilFflonk::G1_BYTES));
    }
    assert(pilfflonk_ctx_fixed_commitments(ctx, 0, 0, fixedBytes.data(), 1) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "n = 1, and Fibonacci has 2 fixed f"));
    assert(pilfflonk_ctx_fixed_commitments(ctx, 0, 1, fixedBytes.data(), 2) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_ctx_fixed_commitments(ctx, 0, 0, nullptr, 2) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_ctx_fixed_commitments(nullptr, 0, 0, fixedBytes.data(), 2) == PILFFLONK_ERR_INVALID_ARGUMENT);

    // [τ]₂ of its SRS, as pilfflonk_srs_g2 writes it.
    uint8_t g2[PilFflonk::SRS_G2_BYTES], expectedG2[PilFflonk::SRS_G2_BYTES];
    assert(pilfflonk_ctx_srs_g2(ctx, 1, g2) == PILFFLONK_OK);
    assert(pilfflonk_srs_g2(&fib.pk->srs(), 1, expectedG2) == PILFFLONK_OK);
    assert(std::equal(g2, g2 + sizeof(g2), expectedG2));
    assert(pilfflonk_ctx_srs_g2(ctx, 2, g2) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_ctx_srs_g2(ctx, 1, nullptr) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_ctx_srs_g2(nullptr, 1, g2) == PILFFLONK_ERR_INVALID_ARGUMENT);
    pilfflonk_ctx_free(ctx);
    pilfflonk_ctx_free(nullptr);
    assert(pilfflonk_ctx_new(nullptr) == nullptr && pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT);
}

// What each broken provingKey/ is refused with, through ProvingKey::load and the C API.
void expectRefused(const KeyFiles &files, int status, const char *message) {
    const KeyDir dir(files);
    assert(pilfflonk_ctx_new(dir.path().c_str()) == nullptr);
    assert(pilfflonk_last_status() == status);
    if (!contains(pilfflonk_last_error(), message)) {
        std::fprintf(stderr, "unexpected message: %s\n", pilfflonk_last_error());
        assert(!"the error does not say what was expected");
    }
}

void testRefusesBrokenKeys() {
    {
        const KeyDir dir;
        fs::remove(dir.path() + "/pilout.globalInfo.json");
        assert(contains(thrown<IoError>([&] { ProvingKey::load(dir.path()); }), "cannot open"));
        assert(pilfflonk_ctx_new(dir.path().c_str()) == nullptr && pilfflonk_last_status() == PILFFLONK_ERR_IO);
        assert(contains(pilfflonk_last_error(), "pilfflonk_ctx_new: globalInfo: cannot open"));
    }
    {
        const KeyDir dir;
        fs::remove(dir.airFile("const"));
        assert(pilfflonk_ctx_new(dir.path().c_str()) == nullptr && pilfflonk_last_status() == PILFFLONK_ERR_IO);
    }
    KeyFiles files;
    files.globalInfo = text(globalInfoJson("stark"));
    expectRefused(files, PILFFLONK_ERR_FORMAT, "backend: must be \"pilfflonk\"");
    files = KeyFiles();
    files.globalInfo = text(globalInfoJson("pilfflonk", 128));
    expectRefused(files, PILFFLONK_ERR_FORMAT, "is not air 0 of airgroup 0 of the globalInfo, Fibonacci of 128 rows");
    files = KeyFiles();
    files.globalInfo = text("{\"backend\": \"pilfflonk\"");
    expectRefused(files, PILFFLONK_ERR_FORMAT, "not valid JSON");
    files = KeyFiles();
    files.constants.pop_back();
    expectRefused(files, PILFFLONK_ERR_FORMAT, ".const has 16383 bytes");
    files = KeyFiles();
    std::fill(files.constants.begin() + 3 * 32, files.constants.begin() + 4 * 32, 0xff);
    expectRefused(files, PILFFLONK_ERR_FORMAT, "Fibonacci.const: the value of row 1, column 1 is not below r");
    files = KeyFiles();
    files.srs = srsBytes(260);
    expectRefused(files, PILFFLONK_ERR_FORMAT, "has 261 coefficients, and the SRS 260 powers");
    files = KeyFiles();
    files.bin.resize(files.bin.size() / 2);
    expectRefused(files, PILFFLONK_ERR_FORMAT, "Fibonacci.bin");

    // The pilfflonkinfo against what the prover derives and supports.
    auto withInfo = [](const std::function<void(json &)> &change) {
        KeyFiles f;
        json info = json::parse(f.info);
        change(info);
        f.info = text(info.dump());
        return f;
    };
    expectRefused(withInfo([](json &j) { j["layout"][5]["degree"] = 262; }), PILFFLONK_ERR_FORMAT,
                  "holds Q with a degree of 262, not the bound of spec A.1, 261");
    expectRefused(withInfo([](json &j) { j["layout"][2]["degree"] = 258; }), PILFFLONK_ERR_FORMAT,
                  "layout f2 has a degree of 258, below the 1 polynomials of 259 coefficients");
    expectRefused(withInfo([](json &j) { j["maxQDegree"] = 1; j["qDeg"] = 2; }), PILFFLONK_ERR_FORMAT,
                  "Q is split");
    expectRefused(withInfo([](json &j) { j["layout"][3]["pols"][0]["id"] = 0; }), PILFFLONK_ERR_FORMAT,
                  "layout f3 packs cm 0, which an earlier f packs too");
}

// ---------------------------------------------------------------------------------------------
// The instance and the opening
// ---------------------------------------------------------------------------------------------

// With the blinding at zero, the prover's polynomials are the oracle's interpolants: its evaluations
// at the oracle's ξ, and Q(ξ), are the oracle's (M14), and every commitment is [p(τ)]₁.
void testUnblindedIsTheOracle() {
    const Fibonacci fib;
    std::unique_ptr<Instance> inst = fib.instance(std::make_unique<ZeroBlinding>());
    assert(inst->nextStage() == 1 && !inst->qCommitted());
    assert(inst->nCommitments(1) == 3 && inst->nCommitments(2) == 1 && inst->nChallenges(1) == 0 &&
           inst->nChallenges(2) == 1);
    const std::vector<G1Point> stage1 = inst->commitStage(1, {});
    assert(stage1.size() == 3 && inst->nextStage() == 2);
    const FrElement stdVc = fr(fib.oracle["stdVc"]);
    const std::vector<G1Point> q = inst->commitQ({stdVc});
    assert(q.size() == 1 && inst->qCommitted());

    const std::vector<LayoutEntry> &layout = fib.air().info().layout;
    for (uint64_t f = 2; f < 6; ++f) {
        const Poly *p = inst->polynomial(f, 0);
        const G1Point &c = f < 5 ? stage1[f - 2] : q[0];
        assert(samePoint(c, g1Times(p->evaluate(testTau()))));
        // Unblinded, a column has at most N coefficients, and Q at most qDeg·N (A.1).
        assert(p->getDegree() < 256);
    }
    assert(inst->polynomial(0, 0) == nullptr && inst->polynomial(2, 1) == nullptr);

    const FrElement xi = fr(fib.oracle["xi"]);
    const Opening opening({inst.get()}, xi);
    assert(eq(opening.xi(), xi)); // powerW = 1
    const std::vector<FrElement> expected = frs(fib.oracle["evals"]);
    assert(sameElements(opening.evaluations(), expected));
    assert(eq(opening.q(0), fr(fib.oracle["q"])));
    assert(contains(thrown<std::invalid_argument>([&] { opening.q(1); }), "no instance 1"));
    assert(layout.size() == 6);
}

// With the blinding on: determinism by seed, the blinding of A.3, the verifier's Q(ξ), inv, invZh,
// and the identity of A.5.
void testBlindedProof() {
    const Fibonacci fib;
    uint8_t seed[32] = {7};
    const Proved a = prove(fib, std::make_unique<BlindingRng>(seed));
    const Proved b = prove(fib, std::make_unique<BlindingRng>(seed));
    seed[0] = 8;
    const Proved c = prove(fib, std::make_unique<BlindingRng>(seed));

    // The same seed, the same proof.
    assert(a.commitments.size() == 4);
    for (uint64_t i = 0; i < 4; ++i) assert(samePoint(a.commitments[i], b.commitments[i]));
    assert(sameElements(a.opening->evaluations(), b.opening->evaluations()));
    assert(samePoint(a.proof.shplonk.w, b.proof.shplonk.w) && samePoint(a.proof.shplonk.wp, b.proof.shplonk.wp));
    assert(eq(a.proof.inv, b.proof.inv) && eq(a.proof.invZh, b.proof.invZh));
    // Another seed: every commitment changes.
    for (uint64_t i = 0; i < 4; ++i) assert(!samePoint(a.commitments[i], c.commitments[i]));
    assert(!samePoint(a.proof.shplonk.w, c.proof.shplonk.w));

    const AirKey &air = fib.air();
    const FrElement w = PilFflonk::rootOfUnity(8);
    for (const Proved *p : {&a, &c}) {
        // Each blinded column is still its column on H, with |O| + 1 more coefficients (A.3).
        for (uint64_t f = 2; f < 5; ++f) {
            const Poly *poly = p->instance->polynomial(f, 0);
            assert(poly->getLength() == 256 + air.blindLength(f) && poly->getDegree() == poly->getLength() - 1);
            const uint64_t id = air.info().layout[f].pols[0].id;
            const uint64_t column = air.info().cmPolsMap[id].stageId;
            auto cell = [&](uint64_t row, uint64_t col) {
                FrElement v;
                E.fr.fromRprLE(v, fib.witness.data() + ((row % 256) * 2 + col) * 32, 32);
                return v;
            };
            for (uint64_t i = 0; i < 256; ++i) {
                const FrElement onH = poly->fastEvaluate(power(w, i));
                if (column < 2) {
                    assert(eq(onH, cell(i, column)));
                } else {
                    // The im pol, the setup's expression 6: l1' − (l1² + l2²).
                    FrElement expected = cell(i + 1, 0);
                    E.fr.sub(expected, expected, E.fr.square(cell(i, 0)));
                    E.fr.sub(expected, expected, E.fr.square(cell(i, 1)));
                    assert(eq(onH, expected));
                }
            }
        }
        // Q within its bound (commitQ checked the rest), and its value at ξ the verifier's.
        assert(p->instance->polynomial(5, 0)->getDegree() < 261);
        const FrElement xi = p->opening->xi();
        assert(eq(xi, p->xiSeed));
        assert(eq(p->opening->q(0), qVerifierAt(fib, p->opening->evaluations(), p->stdVc, p->xiSeed, xi)));
        // invZh = 1/(ξ^N − 1), and inv the inverse of the verifier's denominators' product, both
        // recomputed here: for k = 1 the roots of f are ξ·ω^s, and the denominators Z_{T_i}(y) for
        // i >= 1, then (y − x_m)·Π_{l≠m}(x_m − x_l) for every root.
        assert(eq(E.fr.mul(p->proof.invZh, E.fr.sub(power(xi, 256), E.fr.one())), E.fr.one()));
        const FrElement y = p->proof.shplonk.y;
        FrElement product = E.fr.one();
        const std::vector<LayoutEntry> &layout = air.info().layout;
        std::vector<std::vector<FrElement>> roots;
        for (const LayoutEntry &f : layout) {
            roots.emplace_back();
            for (int64_t s : f.offsets) roots.back().push_back(E.fr.mul(xi, power(w, static_cast<uint64_t>(s))));
        }
        for (uint64_t f = 1; f < layout.size(); ++f) {
            for (const FrElement &x : roots[f]) E.fr.mul(product, product, E.fr.sub(y, x));
        }
        for (const std::vector<FrElement> &T : roots) {
            for (uint64_t m = 0; m < T.size(); ++m) {
                E.fr.mul(product, product, E.fr.sub(y, T[m]));
                for (uint64_t l = 0; l < T.size(); ++l) {
                    if (l != m) E.fr.mul(product, product, E.fr.sub(T[m], T[l]));
                }
            }
        }
        assert(eq(E.fr.mul(p->proof.inv, product), E.fr.one()));
        // The verifier's check of A.5, with τ known.
        assert(shplonkIdentityHolds(fib, *p));
    }
    // A W off by G breaks it.
    Proved tampered = prove(fib, std::make_unique<BlindingRng>(seed));
    tampered.proof.shplonk.w = add(tampered.proof.shplonk.w, g1Times(E.fr.one()));
    assert(!shplonkIdentityHolds(fib, tampered));
}

void testMutatedWitnessIsUnsatisfied() {
    const Fibonacci fib;
    std::vector<uint8_t> mutated = fib.witness;
    mutated[(100 * 2 + 0) * 32] ^= 1; // l1 at row 100
    std::unique_ptr<Instance> inst = fib.instance(std::make_unique<ZeroBlinding>(), mutated);
    inst->commitStage(1, {});
    const std::string message = thrown<UnsatisfiedError>([&] { inst->commitQ({fr(fib.oracle["stdVc"])}); });
    assert(contains(message, "the witness does not satisfy the constraints of Fibonacci"));
    assert(!inst->qCommitted());
    assert(contains(thrown<std::invalid_argument>([&] { Opening({inst.get()}, fr(fib.oracle["xi"])); }),
                    "has not committed Q yet"));
}

void testRefusesArguments() {
    const Fibonacci fib;
    auto blinding = [] { return std::make_unique<ZeroBlinding>(); };
    const std::vector<FrElement> none;
    const std::vector<uint8_t> &w = fib.witness;
    auto newInstance = [&](uint64_t air, const std::vector<uint8_t> &trace, const std::vector<FrElement> &publics,
                           std::unique_ptr<BlindingSource> b) {
        Instance(*fib.pk, 0, air, trace.data(), trace.size(), none, publics, none, std::move(b));
    };
    assert(contains(thrown<std::invalid_argument>([&] { newInstance(1, w, fib.publics, blinding()); }), "no air 1"));
    assert(contains(thrown<std::invalid_argument>([&] { newInstance(0, w, fib.publics, nullptr); }),
                    "no blinding source"));
    assert(contains(thrown<std::invalid_argument>([&] { newInstance(0, w, {}, blinding()); }),
                    "0 publics, and the proof has 3"));
    std::vector<uint8_t> shorter(w.begin(), w.end() - 32);
    assert(contains(thrown<std::invalid_argument>([&] { newInstance(0, shorter, fib.publics, blinding()); }),
                    "the stage-1 witness has 16352 bytes, and 256 rows of 2 columns have 16384"));
    std::vector<uint8_t> nonCanonical = w;
    std::fill(nonCanonical.begin() + (3 * 2 + 1) * 32, nonCanonical.begin() + (3 * 2 + 2) * 32, 0xff);
    assert(contains(thrown<std::invalid_argument>([&] { newInstance(0, nonCanonical, fib.publics, blinding()); }),
                    "row 3, column 1 is not below r"));
    assert(contains(thrown<std::invalid_argument>([&] {
                        Instance(*fib.pk, 0, 0, w.data(), w.size(), {fib.publics[0]}, fib.publics, none, blinding());
                    }),
                    "1 air values for the 0 of stage 1"));

    std::unique_ptr<Instance> inst = fib.instance(blinding());
    const FrElement one = E.fr.one();
    assert(contains(thrown<std::invalid_argument>([&] { inst->commitStage(0, {}); }), "not one of the 1"));
    assert(contains(thrown<std::invalid_argument>([&] { inst->commitStage(2, {}); }), "not one of the 1"));
    assert(contains(thrown<std::invalid_argument>([&] { inst->commitQ({one}); }), "not all committed yet"));
    assert(contains(thrown<std::invalid_argument>([&] { inst->commitStage(1, {one}); }),
                    "1 challenges for the 0 of stage 1"));
    assert(inst->nextStage() == 1);
    inst->commitStage(1, {});
    assert(contains(thrown<std::invalid_argument>([&] { inst->commitStage(1, {}); }), "out of order"));
    assert(contains(thrown<std::invalid_argument>([&] { inst->commitQ({}); }), "0 challenges for the 1 of stage 2"));
    inst->commitQ({fr(fib.oracle["stdVc"])});
    assert(contains(thrown<std::invalid_argument>([&] { inst->commitQ({one}); }), "committed already"));

    assert(contains(thrown<std::invalid_argument>([&] { Opening({}, one); }), "no instances"));
    assert(contains(thrown<std::invalid_argument>([&] { Opening({nullptr}, one); }), "instance 0 is null"));
    // xiSeed = 1: ξ = 1 is in H.
    assert(contains(thrown<std::runtime_error>([&] { Opening({inst.get()}, one); }), "ξ is in H"));
    assert(contains(thrown<std::invalid_argument>([&] { Opening({inst.get()}, E.fr.zero()); }), "xiSeed is zero"));
    // Two instances of one AIR, the second not committed: refused.
    std::unique_ptr<Instance> second = fib.instance(blinding());
    assert(contains(thrown<std::invalid_argument>([&] { Opening({inst.get(), second.get()}, fr(fib.oracle["xi"])); }),
                    "instance 1 has not committed Q yet"));
    // An empty transcript is refused before the opening touches it.
    const Opening opening({inst.get()}, fr(fib.oracle["xi"]));
    Transcript empty;
    assert(contains(thrown<std::invalid_argument>([&] { opening.open(empty); }), "the transcript is empty"));
}

// ---------------------------------------------------------------------------------------------
// The C API
// ---------------------------------------------------------------------------------------------

struct CApiProof {
    std::vector<uint8_t> commitments, evaluations, w, wp, inv, invZh, q;
};

std::vector<uint8_t> scalars(const std::vector<FrElement> &values) {
    std::vector<uint8_t> out(values.size() * 32);
    for (uint64_t i = 0; i < values.size(); ++i) PilFflonk::encodeFr(values[i], out.data() + i * 32);
    return out;
}

// The proof `prove` makes, through the C API.
CApiProof proveThroughTheCApi(const Fibonacci &fib, const uint8_t seed[32]) {
    CApiProof out;
    void *ctx = pilfflonk_ctx_new(fib.dir.path().c_str());
    assert(ctx != nullptr);
    const std::vector<uint8_t> publics = scalars(fib.publics);
    void *inst = pilfflonk_instance_new(ctx, 0, 0, fib.witness.data(), fib.witness.size(), nullptr, 0,
                                        publics.data(), 3, nullptr, 0, seed);
    assert(inst != nullptr);
    void *t = pilfflonk_transcript_new();
    const std::vector<uint8_t> one = scalars({E.fr.one()});
    assert(pilfflonk_transcript_absorb(t, one.data(), 1, PILFFLONK_TRANSCRIPT_FR) == PILFFLONK_OK);
    assert(pilfflonk_transcript_absorb(t, publics.data(), 3, PILFFLONK_TRANSCRIPT_FR) == PILFFLONK_OK);
    out.commitments.resize(4 * 64);
    assert(pilfflonk_commit_stage(inst, 1, nullptr, 0, out.commitments.data(), 3) == PILFFLONK_OK);
    assert(pilfflonk_transcript_absorb(t, out.commitments.data(), 3, PILFFLONK_TRANSCRIPT_G1) == PILFFLONK_OK);
    uint8_t stdVc[32], xiSeed[32];
    assert(pilfflonk_transcript_squeeze(t, stdVc) == PILFFLONK_OK);
    assert(pilfflonk_commit_q(inst, stdVc, 1, out.commitments.data() + 3 * 64, 1) == PILFFLONK_OK);
    assert(pilfflonk_transcript_absorb(t, out.commitments.data() + 3 * 64, 1, PILFFLONK_TRANSCRIPT_G1) ==
           PILFFLONK_OK);
    assert(pilfflonk_transcript_squeeze(t, xiSeed) == PILFFLONK_OK);
    const void *instances[] = {inst};
    void *opening = pilfflonk_opening_new(instances, 1, xiSeed);
    assert(opening != nullptr);
    const uint64_t n = pilfflonk_opening_n_evaluations(opening);
    assert(n == 7 && pilfflonk_last_status() == PILFFLONK_OK);
    out.evaluations.resize(n * 32);
    assert(pilfflonk_opening_evaluations(opening, n, out.evaluations.data()) == PILFFLONK_OK);
    assert(pilfflonk_transcript_absorb(t, out.evaluations.data(), n, PILFFLONK_TRANSCRIPT_FR) == PILFFLONK_OK);
    out.q.resize(32);
    assert(pilfflonk_opening_q(opening, 0, out.q.data()) == PILFFLONK_OK);
    out.w.resize(64), out.wp.resize(64), out.inv.resize(32), out.invZh.resize(32);
    assert(pilfflonk_opening_open(opening, t, out.w.data(), out.wp.data(), out.inv.data(), out.invZh.data()) ==
           PILFFLONK_OK);
    pilfflonk_opening_free(opening);
    pilfflonk_transcript_free(t);
    pilfflonk_instance_free(inst);
    pilfflonk_ctx_free(ctx);
    return out;
}

std::vector<uint8_t> points(const std::vector<G1Point> &list) {
    std::vector<uint8_t> out(list.size() * 64);
    for (uint64_t i = 0; i < list.size(); ++i) PilFflonk::encodeG1(list[i], out.data() + i * 64);
    return out;
}

void testCApi() {
    const Fibonacci fib;
    uint8_t seed[32] = {42};
    const CApiProof api = proveThroughTheCApi(fib, seed);
    const Proved cpp = prove(fib, std::make_unique<BlindingRng>(seed));
    // The C API is the classes, byte for byte.
    assert(api.commitments == points(cpp.commitments));
    assert(api.evaluations == scalars(cpp.opening->evaluations()));
    assert(api.q == scalars({cpp.opening->q(0)}));
    assert(api.w == points({cpp.proof.shplonk.w}) && api.wp == points({cpp.proof.shplonk.wp}));
    assert(api.inv == scalars({cpp.proof.inv}) && api.invZh == scalars({cpp.proof.invZh}));
    assert(proveThroughTheCApi(fib, seed).commitments == api.commitments);

    // Refusals.
    void *ctx = pilfflonk_ctx_new(fib.dir.path().c_str());
    const std::vector<uint8_t> publics = scalars(fib.publics);
    auto instance = [&](const std::vector<uint8_t> &trace, const uint8_t *p, uint64_t nPublics) {
        return pilfflonk_instance_new(ctx, 0, 0, trace.data(), trace.size(), nullptr, 0, p, nPublics, nullptr, 0,
                                      seed);
    };
    assert(pilfflonk_instance_new(nullptr, 0, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr) == nullptr);
    assert(pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(instance(fib.witness, nullptr, 3) == nullptr && contains(pilfflonk_last_error(), "publics is NULL"));
    assert(instance(fib.witness, publics.data(), 2) == nullptr &&
           pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_instance_new(ctx, 0, 1, fib.witness.data(), fib.witness.size(), nullptr, 0, publics.data(), 3,
                                  nullptr, 0, seed) == nullptr &&
           contains(pilfflonk_last_error(), "no air 1"));
    std::vector<uint8_t> nonCanonical = fib.witness;
    std::fill(nonCanonical.begin() + (3 * 2 + 1) * 32, nonCanonical.begin() + (3 * 2 + 2) * 32, 0xff);
    assert(instance(nonCanonical, publics.data(), 3) == nullptr);
    assert(pilfflonk_last_status() == PILFFLONK_ERR_NON_CANONICAL);
    assert(contains(pilfflonk_last_error(), "the stage-1 witness at row 3, column 1 is not below r"));
    std::vector<uint8_t> badPublics = publics;
    std::fill(badPublics.begin() + 32, badPublics.begin() + 64, 0xff);
    assert(instance(fib.witness, badPublics.data(), 3) == nullptr &&
           pilfflonk_last_status() == PILFFLONK_ERR_NON_CANONICAL && contains(pilfflonk_last_error(), "publics[1]"));

    void *inst = instance(fib.witness, publics.data(), 3);
    assert(inst != nullptr);
    uint8_t out[4 * 64];
    const Bytes32 r(R_HEX);
    assert(pilfflonk_commit_stage(inst, 1, nullptr, 0, out, 2) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "n_out = 2, and stage 1 has 3 f to commit"));
    assert(pilfflonk_commit_stage(inst, 1, r.bytes, 1, out, 3) == PILFFLONK_ERR_NON_CANONICAL);
    assert(pilfflonk_commit_stage(inst, 2, nullptr, 0, out, 1) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_commit_stage(nullptr, 1, nullptr, 0, out, 3) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_commit_stage(inst, 1, nullptr, 0, out, 3) == PILFFLONK_OK);
    const Bytes32 xiSeed("0000000000000000000000000000000000000000000000000000000000000002");
    const void *instances[] = {inst};
    assert(pilfflonk_opening_new(instances, 1, xiSeed.bytes) == nullptr);
    assert(pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT && contains(pilfflonk_last_error(), "Q yet"));
    assert(pilfflonk_commit_q(inst, xiSeed.bytes, 1, out, 1) == PILFFLONK_OK);
    assert(pilfflonk_commit_q(inst, xiSeed.bytes, 1, out, 1) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_opening_new(instances, 1, r.bytes) == nullptr &&
           pilfflonk_last_status() == PILFFLONK_ERR_NON_CANONICAL);
    assert(pilfflonk_opening_new(nullptr, 1, xiSeed.bytes) == nullptr &&
           pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT);
    void *opening = pilfflonk_opening_new(instances, 1, xiSeed.bytes);
    assert(opening != nullptr);
    uint8_t evaluations[7 * 32];
    assert(pilfflonk_opening_evaluations(opening, 6, evaluations) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_opening_q(opening, 1, evaluations) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_opening_n_evaluations(nullptr) == 0 && pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT);
    void *t = pilfflonk_transcript_new();
    uint8_t w[64], wp[64], inv[32], invZh[32];
    assert(pilfflonk_opening_open(opening, t, w, wp, inv, invZh) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "the transcript is empty"));
    assert(pilfflonk_opening_open(opening, t, w, nullptr, inv, invZh) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "out_wp is NULL"));
    pilfflonk_transcript_free(t);
    pilfflonk_opening_free(opening);
    pilfflonk_opening_free(nullptr);
    pilfflonk_instance_free(inst);
    pilfflonk_instance_free(nullptr);

    // A mutated witness: PILFFLONK_ERR_UNSATISFIED, and why.
    std::vector<uint8_t> mutated = fib.witness;
    mutated[(100 * 2 + 1) * 32] ^= 1; // l2 at row 100
    inst = instance(mutated, publics.data(), 3);
    assert(pilfflonk_commit_stage(inst, 1, nullptr, 0, out, 3) == PILFFLONK_OK);
    assert(pilfflonk_commit_q(inst, xiSeed.bytes, 1, out, 1) == PILFFLONK_ERR_UNSATISFIED);
    assert(contains(pilfflonk_last_error(), "pilfflonk_commit_q: the witness does not satisfy the constraints"));
    pilfflonk_instance_free(inst);
    pilfflonk_ctx_free(ctx);
}

// ---------------------------------------------------------------------------------------------
// The check (plan M25)
// ---------------------------------------------------------------------------------------------

// The constraints of the Fibonacci's .bin: the pilout's five, then its im pol's.
const char *const FIBONACCI_LINES[] = {
    "fibonacci.pil:24 (l2'-l1)*(1-Fibonacci.LLAST) == 0",
    "fibonacci.pil:27 (l1'-((l1*l1)+(l2*l2)))*(1-Fibonacci.LLAST) == 0",
    "fibonacci.pil:29 Fibonacci.L1*(l2-in1) == 0",
    "fibonacci.pil:30 Fibonacci.L1*(l1-in2) == 0",
    "fibonacci.pil:31 Fibonacci.LLAST*(l1-out) == 0",
    "(Fibonacci.ImPol0 - (l1' - ((l1 * l1) + (l2 * l2)))) == 0",
};
constexpr uint64_t N_CONSTRAINTS = 6;
constexpr uint64_t L1 = 0, L2 = 1;

FrElement cell(const std::vector<uint8_t> &trace, uint64_t row, uint64_t column) {
    return PilFflonk::fromCanonicalFr(trace.data() + (row * 2 + column) * 32);
}

// The trace with its cell (row, column) plus one.
std::vector<uint8_t> plusOne(const std::vector<uint8_t> &trace, uint64_t row, uint64_t column) {
    std::vector<uint8_t> out = trace;
    PilFflonk::encodeFr(E.fr.add(cell(trace, row, column), E.fr.one()), out.data() + (row * 2 + column) * 32);
    return out;
}

using Failures = std::vector<std::pair<uint64_t, uint64_t>>; // (constraint, row)

// Every (constraint, row) a check reports, in order; the check must have kept every failed row.
Failures failures(const std::vector<PilFflonk::ConstraintCheck> &checks) {
    assert(checks.size() == N_CONSTRAINTS);
    Failures out;
    for (uint64_t c = 0; c < checks.size(); ++c) {
        assert(checks[c].rows.size() == checks[c].nFailed);
        for (const PilFflonk::FailedRow &r : checks[c].rows) {
            assert(!E.fr.isZero(r.value));
            out.push_back({c, r.row});
        }
    }
    return out;
}

std::vector<PilFflonk::ConstraintCheck> checkOf(const Fibonacci &fib, const std::vector<uint8_t> &trace,
                                                const std::vector<FrElement> &publics, uint64_t maxRows) {
    Instance inst(*fib.pk, 0, 0, trace.data(), trace.size(), {}, publics, {}, std::make_unique<ZeroBlinding>());
    return inst.check(maxRows);
}

void testCheckReadsTheConstraints() {
    const Fibonacci fib;
    const std::vector<PilFflonk::ParserParams> &constraints = fib.air().bin().constraintsInfoDebug;
    assert(constraints.size() == N_CONSTRAINTS);
    for (uint64_t c = 0; c < N_CONSTRAINTS; ++c) {
        assert(constraints[c].line == FIBONACCI_LINES[c]);
        assert(constraints[c].stage == 1 && constraints[c].firstRow == 0 && constraints[c].lastRow == 256);
        assert(constraints[c].imPol == (c == 5));
    }
    // The generator's witness satisfies them all, im pol's included.
    const std::vector<PilFflonk::ConstraintCheck> checks = checkOf(fib, fib.witness, fib.publics, 10);
    assert(failures(checks).empty());
}

// A mutated cell or public fails exactly the (constraint, row) the oracle says (M14:
// pilfflonk/tests/fibonacci.rs, a_mutated_cell_fails_exactly_the_constraints_and_rows_that_read_it),
// with the values of its numerators there.
void testCheckFindsTheOraclesRows() {
    const Fibonacci fib;
    const struct {
        uint64_t column, row;
        Failures expected;
    } cases[] = {
        {L1, 100, {{0, 100}, {1, 99}, {1, 100}}}, {L2, 100, {{0, 99}, {1, 100}}},
        {L1, 0, {{0, 0}, {1, 0}, {3, 0}}},        {L2, 0, {{1, 0}, {2, 0}}},
        {L1, 255, {{1, 254}, {4, 255}}},          {L2, 255, {{0, 254}}},
        {L1, 1, {{0, 1}, {1, 0}, {1, 1}}},
    };
    for (const auto &m : cases) {
        const std::vector<PilFflonk::ConstraintCheck> checks =
            checkOf(fib, plusOne(fib.witness, m.row, m.column), fib.publics, 256);
        if (failures(checks) != m.expected) {
            std::fprintf(stderr, "column %" PRIu64 ", row %" PRIu64 ": not the oracle's failures\n", m.column, m.row);
            assert(!"the check does not find what the oracle does");
        }
    }

    // l1[100] + 1: (l2' − l1)(1 − LLAST) = −1 at row 100, l1' − next = 1 at row 99, and at row 100
    // l1' − ((l1 + 1)² + l2²) = −(2·l1 + 1).
    const std::vector<PilFflonk::ConstraintCheck> checks =
        checkOf(fib, plusOne(fib.witness, 100, L1), fib.publics, 256);
    const FrElement minusOne = E.fr.neg(E.fr.one());
    const FrElement l1 = cell(fib.witness, 100, L1);
    assert(eq(checks[0].rows[0].value, minusOne));
    assert(eq(checks[1].rows[0].value, E.fr.one()));
    assert(eq(checks[1].rows[1].value, E.fr.neg(E.fr.add(E.fr.add(l1, l1), E.fr.one()))));

    // The publics: out (constraint 4 at N − 1, LLAST·(l1 − out) = −1) and in1 (constraint 2 at 0).
    std::vector<FrElement> publics = fib.publics;
    publics[2] = E.fr.add(publics[2], E.fr.one());
    const std::vector<PilFflonk::ConstraintCheck> out = checkOf(fib, fib.witness, publics, 256);
    assert((failures(out) == Failures{{4, 255}}) && eq(out[4].rows[0].value, minusOne));
    publics = fib.publics;
    publics[0] = E.fr.add(publics[0], E.fr.one());
    assert((failures(checkOf(fib, fib.witness, publics, 256)) == Failures{{2, 0}}));
}

// maxRows keeps the first rows and counts them all.
void testCheckCapsTheRows() {
    const Fibonacci fib;
    // Every l1 + 1: l2' − l1 = −1 on the 255 rows 1 − LLAST keeps, L1·(l1 − in2) = 1 at row 0 and
    // LLAST·(l1 − out) = 1 at row 255; l2 and the im pol's constraint still hold.
    std::vector<uint8_t> trace = fib.witness;
    for (uint64_t row = 0; row < 256; ++row) {
        trace = plusOne(trace, row, L1);
    }
    for (uint64_t maxRows : {uint64_t(0), uint64_t(3), uint64_t(1000)}) {
        const std::vector<PilFflonk::ConstraintCheck> checks = checkOf(fib, trace, fib.publics, maxRows);
        assert(checks[0].nFailed == 255 && checks[2].nFailed == 0 && checks[3].nFailed == 1 &&
               checks[4].nFailed == 1 && checks[5].nFailed == 0);
        for (const PilFflonk::ConstraintCheck &check : checks) {
            assert(check.rows.size() == std::min(check.nFailed, maxRows));
            for (uint64_t j = 0; j < check.rows.size(); ++j) {
                assert(j == 0 || check.rows[j - 1].row < check.rows[j].row);
            }
        }
        for (uint64_t j = 0; j < checks[0].rows.size(); ++j) {
            assert(checks[0].rows[j].row == j);
        }
        if (maxRows > 0) {
            assert(checks[3].rows[0].row == 0 && checks[4].rows[0].row == 255);
        }
    }
}

// Before, between or after the commits, the check finds the same, and the proof is the one without it.
void testCheckLeavesTheProofAsItWas() {
    const Fibonacci fib;
    uint8_t seed[32] = {7};
    const FrElement stdVc = fr(fib.oracle["stdVc"]);
    std::unique_ptr<Instance> plain = fib.instance(std::make_unique<BlindingRng>(seed));
    std::vector<G1Point> expected = plain->commitStage(1, {});
    const std::vector<G1Point> expectedQ = plain->commitQ({stdVc});
    expected.insert(expected.end(), expectedQ.begin(), expectedQ.end());

    std::unique_ptr<Instance> checked = fib.instance(std::make_unique<BlindingRng>(seed));
    assert(failures(checked->check(1)).empty());
    std::vector<G1Point> commitments = checked->commitStage(1, {});
    assert(failures(checked->check(1)).empty());
    const std::vector<G1Point> q = checked->commitQ({stdVc});
    assert(failures(checked->check(1)).empty());
    commitments.insert(commitments.end(), q.begin(), q.end());
    assert(commitments.size() == expected.size());
    for (uint64_t i = 0; i < commitments.size(); ++i) {
        assert(samePoint(commitments[i], expected[i]));
    }

    // A mutated witness: the same rows after its stage 1 is committed.
    const std::vector<uint8_t> mutated = plusOne(fib.witness, 100, L1);
    std::unique_ptr<Instance> inst = fib.instance(std::make_unique<ZeroBlinding>(), mutated);
    inst->commitStage(1, {});
    assert((failures(inst->check(10)) == Failures{{0, 100}, {1, 99}, {1, 100}}));
}

// The offset in a .bin of word `word` (0 stage, 1 destId, 2 firstRow, 3 lastRow) of the entry of
// constraint c in section 2 (the format: setup/pilfflonk/src/bytecode.rs).
uint64_t constraintWord(const std::vector<uint8_t> &bin, uint64_t c, uint64_t word) {
    uint64_t size1 = 0;
    std::memcpy(&size1, bin.data() + 12 + 4, sizeof(size1)); // after "chps", version, nSections, id 1
    uint64_t at = 12 + 12 + size1 + 12 + 16;                  // section 2's entries, after its four counts
    for (uint64_t i = 0; i < c; ++i) {
        at += 10 * 4; // stage … argsOffset, imPol
        at = std::find(bin.begin() + at, bin.end(), uint8_t(0)) - bin.begin() + 1;
    }
    return at + 4 * word;
}

void setWord(std::vector<uint8_t> &bin, uint64_t at, uint32_t value) {
    std::memcpy(bin.data() + at, &value, sizeof(value));
}

void testCheckRefusals() {
    // A constraint of stage 2 (whose columns the std's hints compute, M30): refused before anything.
    {
        KeyFiles files;
        for (uint64_t c = 0; c < N_CONSTRAINTS; ++c) {
            uint32_t stage = 0, lastRow = 0;
            std::memcpy(&stage, files.bin.data() + constraintWord(files.bin, c, 0), sizeof(stage));
            std::memcpy(&lastRow, files.bin.data() + constraintWord(files.bin, c, 3), sizeof(lastRow));
            assert(stage == 1 && lastRow == 256);
        }
        setWord(files.bin, constraintWord(files.bin, 1, 0), 2);
        const KeyDir dir(files);
        const std::unique_ptr<ProvingKey> pk = ProvingKey::load(dir.path());
        const std::vector<uint8_t> witness = readBytes(fixture("Fibonacci.witness.bin"));
        const std::vector<FrElement> publics = frs(json::parse(readBytes(fixture("Fibonacci.oracle.json")))["publics"]);
        Instance inst(*pk, 0, 0, witness.data(), witness.size(), {}, publics, {}, std::make_unique<ZeroBlinding>());
        const std::string message = thrown<std::invalid_argument>([&] { inst.check(10); });
        assert(contains(message, std::string("constraint 1 (") + FIBONACCI_LINES[1] + ") is of stage 2"));
        assert(contains(message, "(plan M30)"));
    }
    // A constraint's rows beyond the trace: the key is refused.
    KeyFiles files;
    setWord(files.bin, constraintWord(files.bin, 4, 3), 257);
    expectRefused(files, PILFFLONK_ERR_FORMAT,
                  "Fibonacci: .bin: constraint 4 holds on the rows 0 <= i < 257, and the trace has 256");
}

void testCheckCApi() {
    const Fibonacci fib;
    void *ctx = pilfflonk_ctx_new(fib.dir.path().c_str());
    assert(ctx != nullptr);
    uint64_t n = 0;
    assert(pilfflonk_ctx_n_constraints(ctx, 0, 0, &n) == PILFFLONK_OK && n == N_CONSTRAINTS);
    assert(pilfflonk_ctx_n_constraints(ctx, 0, 1, &n) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_ctx_n_constraints(nullptr, 0, 0, &n) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_ctx_n_constraints(ctx, 0, 0, nullptr) == PILFFLONK_ERR_INVALID_ARGUMENT);

    uint64_t stage = 0, firstRow = 0, lastRow = 0, lineLen = 0;
    uint32_t imPol = 2;
    for (uint64_t c = 0; c < N_CONSTRAINTS; ++c) {
        assert(pilfflonk_ctx_constraint(ctx, 0, 0, c, &stage, &firstRow, &lastRow, &imPol, &lineLen) == PILFFLONK_OK);
        assert(stage == 1 && firstRow == 0 && lastRow == 256 && imPol == (c == 5 ? 1 : 0));
        assert(lineLen == std::strlen(FIBONACCI_LINES[c]));
        std::vector<uint8_t> line(lineLen);
        assert(pilfflonk_ctx_constraint_line(ctx, 0, 0, c, line.data(), lineLen) == PILFFLONK_OK);
        assert(std::string(line.begin(), line.end()) == FIBONACCI_LINES[c]);
        assert(pilfflonk_ctx_constraint_line(ctx, 0, 0, c, line.data(), lineLen - 1) == PILFFLONK_ERR_INVALID_ARGUMENT);
        assert(contains(pilfflonk_last_error(), "and the line of constraint " + std::to_string(c) + " has " +
                                                    std::to_string(lineLen) + " bytes"));
    }
    assert(pilfflonk_ctx_constraint(ctx, 0, 0, 6, &stage, &firstRow, &lastRow, &imPol, &lineLen) ==
           PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "Fibonacci has no constraint 6, of 6"));
    assert(pilfflonk_ctx_constraint(ctx, 1, 0, 0, &stage, &firstRow, &lastRow, &imPol, &lineLen) ==
           PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_ctx_constraint(ctx, 0, 0, 0, &stage, &firstRow, nullptr, &imPol, &lineLen) ==
           PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "last_row is NULL"));
    assert(pilfflonk_ctx_constraint_line(ctx, 0, 0, 6, nullptr, 0) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_ctx_constraint_line(ctx, 0, 0, 0, nullptr, 3) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_ctx_constraint_line(nullptr, 0, 0, 0, nullptr, 0) == PILFFLONK_ERR_INVALID_ARGUMENT);

    // l1[100] + 1, two rows kept per constraint: what Instance::check finds, the rest zeroed.
    const std::vector<uint8_t> mutated = plusOne(fib.witness, 100, L1);
    const std::vector<uint8_t> publics = scalars(fib.publics);
    uint8_t seed[32] = {1};
    void *inst = pilfflonk_instance_new(ctx, 0, 0, mutated.data(), mutated.size(), nullptr, 0, publics.data(), 3,
                                        nullptr, 0, seed);
    assert(inst != nullptr);
    constexpr uint64_t MAX_ROWS = 2;
    std::vector<uint64_t> nFailed(N_CONSTRAINTS, 99), rows(N_CONSTRAINTS * MAX_ROWS, 99);
    std::vector<uint8_t> values(N_CONSTRAINTS * MAX_ROWS * 32, 0xff);
    assert(pilfflonk_check(inst, MAX_ROWS, N_CONSTRAINTS, nFailed.data(), rows.data(), values.data()) == PILFFLONK_OK);
    assert(pilfflonk_last_error()[0] == '\0');
    const std::vector<PilFflonk::ConstraintCheck> expected = checkOf(fib, mutated, fib.publics, MAX_ROWS);
    for (uint64_t c = 0; c < N_CONSTRAINTS; ++c) {
        assert(nFailed[c] == expected[c].nFailed);
        for (uint64_t j = 0; j < MAX_ROWS; ++j) {
            const uint64_t e = c * MAX_ROWS + j;
            const std::vector<uint8_t> value(values.begin() + e * 32, values.begin() + (e + 1) * 32);
            if (j < expected[c].rows.size()) {
                assert(rows[e] == expected[c].rows[j].row && value == scalars({expected[c].rows[j].value}));
            } else {
                assert(rows[e] == 0 && value == std::vector<uint8_t>(32, 0));
            }
        }
    }
    assert((nFailed == std::vector<uint64_t>{1, 2, 0, 0, 0, 0}));
    assert(rows[0] == 100 && rows[2] == 99 && rows[3] == 100);
    // Counting only.
    std::fill(nFailed.begin(), nFailed.end(), 99);
    assert(pilfflonk_check(inst, 0, N_CONSTRAINTS, nFailed.data(), nullptr, nullptr) == PILFFLONK_OK);
    assert((nFailed == std::vector<uint64_t>{1, 2, 0, 0, 0, 0}));
    // The instance still proves as one the check never saw: its Q is not a polynomial.
    uint8_t out[4 * 64];
    assert(pilfflonk_commit_stage(inst, 1, nullptr, 0, out, 3) == PILFFLONK_OK);
    const Bytes32 stdVc("0000000000000000000000000000000000000000000000000000000000000002");
    assert(pilfflonk_commit_q(inst, stdVc.bytes, 1, out, 1) == PILFFLONK_ERR_UNSATISFIED);

    // Refusals.
    assert(pilfflonk_check(nullptr, 1, N_CONSTRAINTS, nFailed.data(), rows.data(), values.data()) ==
           PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_check(inst, 1, 5, nFailed.data(), rows.data(), values.data()) == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "pilfflonk_check: n_constraints = 5, and Fibonacci has 6"));
    assert(pilfflonk_check(inst, 1, N_CONSTRAINTS, nullptr, rows.data(), values.data()) ==
           PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "out_n_failed is NULL"));
    assert(pilfflonk_check(inst, 1, N_CONSTRAINTS, nFailed.data(), nullptr, values.data()) ==
           PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(pilfflonk_check(inst, 1, N_CONSTRAINTS, nFailed.data(), rows.data(), nullptr) ==
           PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "out_values is NULL"));
    assert(pilfflonk_check(inst, uint64_t(1) << 60, N_CONSTRAINTS, nFailed.data(), rows.data(), values.data()) ==
           PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "scalars exceed 2^64 bytes"));
    pilfflonk_instance_free(inst);
    pilfflonk_ctx_free(ctx);
}

} // namespace

void runProverTests() {
    testBlindingRng();
    testLoadsTheFibonaccisKey();
    testRefusesBrokenKeys();
    testUnblindedIsTheOracle();
    testBlindedProof();
    testMutatedWitnessIsUnsatisfied();
    testRefusesArguments();
    testCApi();
    testCheckReadsTheConstraints();
    testCheckFindsTheOraclesRows();
    testCheckCapsTheRows();
    testCheckLeavesTheProofAsItWas();
    testCheckRefusals();
    testCheckCApi();
}

} // namespace PilFflonkTest
