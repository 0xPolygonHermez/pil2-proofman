// The data-driven step evaluator against pil1's generated code.
//
// Both compute the same stage expressions for the reference AIR: the generated
// functions in chelpers/ were compiled from the same source the .fflonkinfo
// carries as data. Running both over identical inputs and comparing every
// output buffer is what establishes that a prover driven by data reproduces one
// driven by generated C++ -- which is the prerequisite for substituting pil2's
// bytecode later.
#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "chelpers/pilfflonk_steps.hpp"
#include "fflonk_info.hpp"
#include "json_steps.hpp"
#include "utils.hpp"

using PilFflonk::FrEl;
using PilFflonk::JsonSteps;

namespace {

const std::string FIXTURES = "../tests/fixtures/reference/";
constexpr uint64_t N = 256;

/// The extended domain the quotient is evaluated over.
///
/// The prover sizes it as `nBits + nBitsZK + ceil(log2(qDeg + 1))` -- the
/// blinding bits count too, which is easy to miss: taking it as `N * (qDeg+1)`
/// gives a buffer half the size and reads off the end.
constexpr uint64_t NEXT = 1ULL << (8 + 1 + 2);

/// Buffers for one run, sized from the AIR.
struct Buffers {
    std::vector<FrEl> cm1, cm2, cm3, tmpExp, constPols, challenges, publics, x, q;
    /// The extended-domain counterparts, which the quotient step reads.
    std::vector<FrEl> cm1x, cm2x, cm3x, constx, xx;
    /// Literals the generated code hoists out of the expressions. The
    /// data-driven evaluator reads them inline and needs none, but the
    /// generated path dereferences this.
    std::vector<FrEl> constValues;

    explicit Buffers(FflonkInfo::FflonkInfo &info) {
        cm1.resize(N * info.mapSectionsN.section[FflonkInfo::cm1_n]);
        cm2.resize(N * std::max<uint64_t>(1, info.mapSectionsN.section[FflonkInfo::cm2_n]));
        cm3.resize(N * std::max<uint64_t>(1, info.mapSectionsN.section[FflonkInfo::cm3_n]));
        tmpExp.resize(N * std::max<uint64_t>(1, info.mapSectionsN.section[FflonkInfo::tmpExp_n]));
        constPols.resize(N * info.nConstants);
        challenges.resize(8);
        publics.resize(std::max<uint64_t>(1, info.nPublics));
        x.resize(N);
        q.resize(NEXT * std::max<uint64_t>(1, info.mapSectionsN.section[FflonkInfo::q_2ns]));

        cm1x.resize(NEXT * std::max<uint64_t>(1, info.mapSectionsN.section[FflonkInfo::cm1_2ns]));
        cm2x.resize(NEXT * std::max<uint64_t>(1, info.mapSectionsN.section[FflonkInfo::cm2_2ns]));
        cm3x.resize(NEXT * std::max<uint64_t>(1, info.mapSectionsN.section[FflonkInfo::cm3_2ns]));
        constx.resize(NEXT * info.nConstants);
        xx.resize(NEXT);

        PilFflonkSteps compiled;
        constValues.resize(std::max<uint64_t>(1, compiled.getNumConstValues()));
    }

    PilFflonkStepsParams params() {
        PilFflonkStepsParams p{};
        p.cm1_n = cm1.data();
        p.cm2_n = cm2.data();
        p.cm3_n = cm3.data();
        p.tmpExp_n = tmpExp.data();
        p.const_n = constPols.data();
        p.challenges = challenges.data();
        p.publicInputs = publics.data();
        p.x_n = x.data();
        p.q_2ns = q.data();
        p.cm1_2ns = cm1x.data();
        p.cm2_2ns = cm2x.data();
        p.cm3_2ns = cm3x.data();
        p.const_2ns = constx.data();
        p.x_2ns = xx.data();
        p.constValues = constValues.data();
        return p;
    }
};

/// Deterministic pseudo-random field elements, so both runs see identical
/// inputs and a failure is reproducible.
void fill(std::vector<FrEl> &buf, uint64_t seed) {
    auto &E = AltBn128::Engine::engine;
    std::mt19937_64 rng(seed);
    for (auto &e : buf) {
        e = E.fr.set(rng() % 1000000007ULL);
    }
}

void fillInputs(Buffers &b, uint64_t seed) {
    fill(b.cm1, seed);
    fill(b.constPols, seed + 1);
    fill(b.challenges, seed + 2);
    fill(b.publics, seed + 3);
    fill(b.x, seed + 4);
}

/// Compare buffers element-wise: the field element has no operator==, and
/// comparing the raw bytes would be wrong for values held in different but
/// equivalent representations.
::testing::AssertionResult same(const std::vector<FrEl> &a, const std::vector<FrEl> &b, const char *what) {
    auto &E = AltBn128::Engine::engine;
    if (a.size() != b.size()) {
        return ::testing::AssertionFailure() << what << ": sizes differ (" << a.size() << " vs " << b.size() << ")";
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!E.fr.eq(a[i], b[i])) {
            return ::testing::AssertionFailure()
                   << what << ": element " << i << " differs (" << E.fr.toString(a[i]) << " vs "
                   << E.fr.toString(b[i]) << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

} // namespace

// step2prev computes the plookup helper columns. Comparing the whole of every
// buffer it can write catches a wrong section as readily as a wrong value.
TEST(JSON_STEPS, reproducesTheGeneratedStep2Prev) {
    FflonkInfo::FflonkInfo fflonkInfo(AltBn128::Engine::engine, FIXTURES + "pilfflonk.fflonkinfo.json");

    JsonSteps steps(fflonkInfo, fflonkInfo.step2prev, JsonSteps::Section::First, PilFflonk::Domain::Base);
    ASSERT_GT(steps.size(), 0u);

    Buffers generated(fflonkInfo), evaluated(fflonkInfo);
    fillInputs(generated, 42);
    fillInputs(evaluated, 42);

    auto genParams = generated.params();
    auto evalParams = evaluated.params();

    PilFflonkSteps compiled;
    for (uint64_t i = 0; i < N; i++) {
        compiled.step2prev_first(AltBn128::Engine::engine, genParams, i);
    }
    for (uint64_t i = 0; i < N; i++) {
        steps.run(AltBn128::Engine::engine, evalParams, i, N);
    }

    EXPECT_TRUE(same(generated.tmpExp, evaluated.tmpExp, "tmpExp differs"));
    EXPECT_TRUE(same(generated.cm3, evaluated.cm3, "cm3 differs"));
    EXPECT_TRUE(same(generated.cm2, evaluated.cm2, "cm2 differs"));
    EXPECT_TRUE(same(generated.cm1, evaluated.cm1, "the evaluator wrote to its inputs"));
}

// step3 runs after the grand products, reading what stage 2 wrote.
TEST(JSON_STEPS, reproducesTheGeneratedStep3) {
    FflonkInfo::FflonkInfo fflonkInfo(AltBn128::Engine::engine, FIXTURES + "pilfflonk.fflonkinfo.json");

    JsonSteps steps(fflonkInfo, fflonkInfo.step3, JsonSteps::Section::First, PilFflonk::Domain::Base);

    Buffers generated(fflonkInfo), evaluated(fflonkInfo);
    fillInputs(generated, 7);
    fillInputs(evaluated, 7);
    // stage 3 reads stage 2's output, so seed those too.
    fill(generated.cm2, 99);
    fill(evaluated.cm2, 99);
    fill(generated.cm3, 100);
    fill(evaluated.cm3, 100);

    auto genParams = generated.params();
    auto evalParams = evaluated.params();

    PilFflonkSteps compiled;
    for (uint64_t i = 0; i < N; i++) {
        compiled.step3_first(AltBn128::Engine::engine, genParams, i);
    }
    for (uint64_t i = 0; i < N; i++) {
        steps.run(AltBn128::Engine::engine, evalParams, i, N);
    }

    EXPECT_TRUE(same(generated.tmpExp, evaluated.tmpExp, "tmpExp"));
    EXPECT_TRUE(same(generated.cm3, evaluated.cm3, "cm3"));
}

// A primed access on the final row must read row zero. Getting this wrong reads
// past the trace, and only shows up at one row in 256.
TEST(JSON_STEPS, wrapsPrimedAccessesAtTheBoundary) {
    FflonkInfo::FflonkInfo fflonkInfo(AltBn128::Engine::engine, FIXTURES + "pilfflonk.fflonkinfo.json");

    JsonSteps steps(fflonkInfo, fflonkInfo.step2prev, JsonSteps::Section::First, PilFflonk::Domain::Base);

    Buffers generated(fflonkInfo), evaluated(fflonkInfo);
    fillInputs(generated, 5);
    fillInputs(evaluated, 5);

    auto genParams = generated.params();
    auto evalParams = evaluated.params();

    PilFflonkSteps compiled;
    compiled.step2prev_first(AltBn128::Engine::engine, genParams, N - 1);
    steps.run(AltBn128::Engine::engine, evalParams, N - 1, N);

    EXPECT_TRUE(same(generated.tmpExp, evaluated.tmpExp, "the last row disagrees, so the wrap differs"));
    EXPECT_TRUE(same(generated.cm3, evaluated.cm3, "cm3"));
}

// step42ns runs over the extended domain, reading the _2ns buffers. varPolMap
// records only the base section, so the domain is what decides which of each
// pair a polynomial resolves to -- getting that wrong reads the trace instead
// of its extension, silently.
TEST(JSON_STEPS, reproducesTheGeneratedStep42ns) {
    FflonkInfo::FflonkInfo fflonkInfo(AltBn128::Engine::engine, FIXTURES + "pilfflonk.fflonkinfo.json");

    JsonSteps steps(fflonkInfo, fflonkInfo.step42ns, JsonSteps::Section::First, PilFflonk::Domain::Extended);

    Buffers generated(fflonkInfo), evaluated(fflonkInfo);
    fillInputs(generated, 11);
    fillInputs(evaluated, 11);
    fill(generated.cm1x, 12); fill(evaluated.cm1x, 12);
    fill(generated.cm2x, 13); fill(evaluated.cm2x, 13);
    fill(generated.cm3x, 14); fill(evaluated.cm3x, 14);
    fill(generated.constx, 15); fill(evaluated.constx, 15);
    fill(generated.xx, 16); fill(evaluated.xx, 16);

    auto genParams = generated.params();
    auto evalParams = evaluated.params();

    PilFflonkSteps compiled;
    compiled.setConstValues(AltBn128::Engine::engine, genParams);

    for (uint64_t i = 0; i < NEXT; i++) {
        compiled.step42ns_first(AltBn128::Engine::engine, genParams, i);
    }
    // On the extended domain the next trace row is `NEXT / N` rows away.
    for (uint64_t i = 0; i < NEXT; i++) {
        steps.run(AltBn128::Engine::engine, evalParams, i, NEXT, NEXT / N);
    }

    EXPECT_TRUE(same(generated.q, evaluated.q, "q_2ns"));
    EXPECT_TRUE(same(generated.cm1x, evaluated.cm1x, "cm1_2ns must be untouched"));
}
