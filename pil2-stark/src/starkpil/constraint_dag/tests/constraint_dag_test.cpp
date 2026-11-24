// The constraint DAG reader, checker and evaluator against the definition: Horner(roots, vc) * 1/(xi^N - 1).
#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "constraint_dag.hpp"

namespace
{

using F3 = Goldilocks3;
using GE = Goldilocks::Element;

PolMap chal(const std::string &name, uint64_t stage, uint64_t stageId)
{
    PolMap p{};
    p.name = name;
    p.stage = stage;
    p.stageId = stageId;
    p.dim = FIELD_EXTENSION;
    return p;
}

PolMap valueMap(uint64_t stage)
{
    PolMap p{};
    p.stage = stage;
    p.dim = stage == 1 ? 1 : FIELD_EXTENSION;
    return p;
}

EvMap ev(EvMap::eType type, uint64_t id, int64_t prime, uint64_t commitId = 0)
{
    EvMap e{};
    e.type = type;
    e.id = id;
    e.prime = prime;
    e.commitId = commitId;
    return e;
}

// nBits=3; evMap: cm@0, cm@-1, const@0, custom@0; one public; air values {stage 1, stage 2}; proof/airgroup values stage 1.
StarkInfo makeInfo()
{
    StarkInfo si;
    si.starkStruct.nBits = 3;
    si.nStages = 2;
    si.nPublics = 1;
    si.challengesMap = {chal("std_alpha", 2, 0), chal("std_vc", 3, 0), chal("std_xi", 4, 0)};
    si.evMap = {ev(EvMap::cm, 0, 0), ev(EvMap::cm, 0, -1), ev(EvMap::_const, 0, 0), ev(EvMap::custom, 0, 0, 0)};
    si.airValuesMap = {valueMap(1), valueMap(2)};
    si.proofValuesMap = {valueMap(1)};
    si.airgroupValuesMap = {valueMap(1)};
    si.boundaries = {Boundary{"everyRow", 0, 0}};
    return si;
}

// Every kind; root i's value is c0 = E2 + E0*E1, c1 = E3*alpha - p + a0*a1, c2 = -(7*pv*g) - vc.
const char *NODES =
    "[[5,0,0,0],[5,0,0,-1],[5,1,0,0],[5,2,0,0,0],[6,3,0],[6,2,0],[7,0],[9,0],[9,1],[8,0],[10,0],[4,\"7\"],"
    "[2,0,1],[0,2,12],[2,3,5],[1,14,6],[2,7,8],[0,15,16],[2,9,10],[2,11,18],[3,19],[1,20,4]]";
const char *ROOTS = "[[13,0,\"c0\"],[17,0,\"c1\"],[21,0,\"c2\"]]";

json makeJson(const std::string &nodes = NODES, const std::string &roots = ROOTS, int version = 1)
{
    return json::parse("{\"version\":" + std::to_string(version) + ",\"field\":\"gl\",\"extDim\":3,\"nodes\":" + nodes +
                       ",\"roots\":" + roots + "}");
}

struct Values
{
    std::vector<GE> evals, challenges, publics, proofValues, airValues, airgroupValues;
    GE zi[3];
    Values()
    {
        std::mt19937_64 rng(7);
        auto fill = [&](std::vector<GE> &v, size_t n) {
            v.resize(n);
            for (auto &e : v) e = Goldilocks::fromU64(rng() % GOLDILOCKS_PRIME);
        };
        fill(evals, 4 * 3);
        fill(challenges, 3 * 3);
        fill(publics, 1);
        fill(proofValues, 1);
        fill(airValues, 1 + 3);
        fill(airgroupValues, 1);
        for (auto &e : zi) e = Goldilocks::fromU64(rng() % GOLDILOCKS_PRIME);
    }
    DagInputs inputs() const
    {
        return DagInputs{evals.data(), challenges.data(), publics.data(), proofValues.data(), airValues.data(),
                         airgroupValues.data(), {zi[0], zi[1], zi[2]}};
    }
};

// Goldilocks3::Element is a raw array: carry values in a struct and cast at the call.
struct V3
{
    GE c[3];
    F3::Element &r() { return reinterpret_cast<F3::Element &>(c); }
    const F3::Element &r() const { return reinterpret_cast<const F3::Element &>(c); }
};

V3 e3(const GE *p) { return V3{{p[0], p[1], p[2]}}; }
V3 base(const GE &x) { return V3{{x, Goldilocks::zero(), Goldilocks::zero()}}; }
V3 mul(const V3 &a, const V3 &b) { V3 o; F3::mul(o.r(), a.r(), b.r()); return o; }
V3 add(const V3 &a, const V3 &b) { V3 o; F3::add(o.r(), a.r(), b.r()); return o; }
V3 sub(const V3 &a, const V3 &b) { V3 o; F3::sub(o.r(), a.r(), b.r()); return o; }

V3 direct(const Values &v)
{
    V3 E0 = e3(&v.evals[0]), E1 = e3(&v.evals[3]), E2 = e3(&v.evals[6]), E3 = e3(&v.evals[9]);
    V3 alpha = e3(&v.challenges[0]), vc = e3(&v.challenges[3]);
    V3 p = base(v.publics[0]), a0 = base(v.airValues[0]), a1 = e3(&v.airValues[1]);
    V3 pv = base(v.proofValues[0]), g = base(v.airgroupValues[0]);
    V3 zero = base(Goldilocks::zero());
    V3 c0 = add(E2, mul(E0, E1));
    V3 c1 = add(sub(mul(E3, alpha), p), mul(a0, a1));
    V3 c2 = sub(sub(zero, mul(base(Goldilocks::fromU64(7)), mul(pv, g))), vc);
    V3 acc = add(mul(vc, c0), c1);
    acc = add(mul(vc, acc), c2);
    return mul(acc, e3(v.zi));
}

V3 evalTo(const ConstraintDag &dag, const StarkInfo &si, const Values &v)
{
    V3 o;
    evalConstraintDag(dag, si, v.inputs(), o.c);
    return o;
}

bool same(const V3 &a, const V3 &b)
{
    for (int k = 0; k < 3; k++)
        if (Goldilocks::toU64(a.c[k]) != Goldilocks::toU64(b.c[k])) return false;
    return true;
}

bool readAndCheck(const json &j, const StarkInfo &si, ConstraintDag &dag, std::string &err)
{
    return readConstraintDag(j, dag, err) && checkConstraintDag(dag, si, err);
}

TEST(ConstraintDag, MatchesDirectFormula)
{
    // zi stands for 1/(xi^8 - 1); the evaluator multiplies by whatever zi it is given.
    const StarkInfo si = makeInfo();
    ConstraintDag dag;
    std::string err;
    ASSERT_TRUE(readAndCheck(makeJson(), si, dag, err)) << err;
    Values v;
    EXPECT_TRUE(same(evalTo(dag, si, v), direct(v)));
}

TEST(ConstraintDag, ZiIsOneOverXiNMinusOne)
{
    const StarkInfo si = makeInfo();
    ConstraintDag dag;
    std::string err;
    ASSERT_TRUE(readAndCheck(makeJson(), si, dag, err)) << err;
    Values v;
    V3 xi = e3(&v.challenges[6]), p = base(Goldilocks::one()), zInv;
    F3::pow(p.r(), xi.r(), 8);
    p = sub(p, base(Goldilocks::one()));
    F3::inv(zInv.r(), p.r());
    for (int k = 0; k < 3; k++) v.zi[k] = zInv.c[k];
    EXPECT_TRUE(same(evalTo(dag, si, v), direct(v)));
}

TEST(ConstraintDag, DetectsChangedEval)
{
    const StarkInfo si = makeInfo();
    ConstraintDag dag;
    std::string err;
    ASSERT_TRUE(readAndCheck(makeJson(), si, dag, err)) << err;
    Values v;
    V3 before = evalTo(dag, si, v);
    v.evals[4] = v.evals[4] + Goldilocks::one();
    V3 after = evalTo(dag, si, v);
    EXPECT_FALSE(same(before, after));
    EXPECT_TRUE(same(after, direct(v)));
}

TEST(ConstraintDag, RejectsForwardReference)
{
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readConstraintDag(makeJson("[[2,0,1],[5,0,0,0],[5,0,0,-1]]", "[[0,0,\"x\"]]"), dag, err));
    EXPECT_NE(err.find("forward"), std::string::npos) << err;
}

TEST(ConstraintDag, RejectsUnknownOpcode)
{
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readConstraintDag(makeJson("[[11,0]]", "[[0,0,\"x\"]]"), dag, err));
    EXPECT_FALSE(err.empty());
}

TEST(ConstraintDag, RejectsWrongArity)
{
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readConstraintDag(makeJson("[[5,0,0,0],[0,0]]", "[[1,0,\"x\"]]"), dag, err));
    EXPECT_FALSE(err.empty());
}

TEST(ConstraintDag, RejectsNonCanonicalConst)
{
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readConstraintDag(makeJson("[[4,\"18446744069414584321\"]]", "[[0,0,\"x\"]]"), dag, err));
    EXPECT_FALSE(err.empty());
}

TEST(ConstraintDag, RejectsUnsortedChildren)
{
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readConstraintDag(makeJson("[[5,0,0,0],[5,0,0,-1],[2,1,0]]", "[[2,0,\"x\"]]"), dag, err));
    EXPECT_NE(err.find("sorted"), std::string::npos) << err;
}

TEST(ConstraintDag, RejectsWrongVersion)
{
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readConstraintDag(makeJson(NODES, ROOTS, 2), dag, err));
    EXPECT_NE(err.find("version"), std::string::npos) << err;
}

TEST(ConstraintDag, RejectsUnknownColumn)
{
    const StarkInfo si = makeInfo();
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readAndCheck(makeJson("[[5,0,9,0]]", "[[0,0,\"x\"]]"), si, dag, err)); // no evMap entry for cm id 9
    EXPECT_FALSE(err.empty());
}

TEST(ConstraintDag, RejectsUnknownChallenge)
{
    const StarkInfo si = makeInfo();
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readAndCheck(makeJson("[[6,2,5]]", "[[0,0,\"x\"]]"), si, dag, err));
    EXPECT_FALSE(err.empty());
}

TEST(ConstraintDag, RejectsMissingStdVc)
{
    StarkInfo si = makeInfo();
    si.challengesMap[1].name = "std_other";
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readAndCheck(makeJson(), si, dag, err));
    EXPECT_NE(err.find("std_vc"), std::string::npos) << err;
}

TEST(ConstraintDag, RejectsNonEveryRowRoot)
{
    const StarkInfo si = makeInfo();
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readAndCheck(makeJson(NODES, "[[13,0,\"c0\"],[17,1,\"c1\"],[21,0,\"c2\"]]"), si, dag, err));
    EXPECT_NE(err.find("everyRow"), std::string::npos) << err;
}

TEST(ConstraintDag, RejectsDuplicateEvEntry)
{
    StarkInfo si = makeInfo();
    si.evMap.push_back(si.evMap[0]);
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readAndCheck(makeJson(), si, dag, err));
    EXPECT_NE(err.find("duplicate"), std::string::npos) << err;
}

TEST(ConstraintDag, RejectsDuplicateChallenge)
{
    StarkInfo si = makeInfo();
    si.challengesMap.push_back(si.challengesMap[0]);
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readAndCheck(makeJson(), si, dag, err));
    EXPECT_NE(err.find("duplicate"), std::string::npos) << err;
}

TEST(ConstraintDag, RejectsPublicOutOfRange)
{
    const StarkInfo si = makeInfo(); // nPublics = 1
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readAndCheck(makeJson("[[7,1]]", "[[0,0,\"x\"]]"), si, dag, err));
    EXPECT_NE(err.find("public"), std::string::npos) << err;
}

// Dimension corruption: the map says ext for a stage-1 value, or base for a later-stage one.
TEST(ConstraintDag, RejectsWrongDimensionStage1ValueTypedExt)
{
    StarkInfo si = makeInfo();
    si.airValuesMap[0].dim = FIELD_EXTENSION;
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readAndCheck(makeJson(), si, dag, err));
    EXPECT_NE(err.find("dimension"), std::string::npos) << err;
}

TEST(ConstraintDag, RejectsWrongDimensionLaterStageValueTypedBase)
{
    StarkInfo si = makeInfo();
    si.airValuesMap[1].dim = 1;
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readAndCheck(makeJson(), si, dag, err));
    EXPECT_NE(err.find("dimension"), std::string::npos) << err;
}

// A col node is ext whatever its column's dim (evals at xi are ext), so the ext slot to corrupt is a challenge.
TEST(ConstraintDag, RejectsWrongDimensionChallengeTypedBase)
{
    StarkInfo si = makeInfo();
    si.challengesMap[0].dim = 1;
    ConstraintDag dag;
    std::string err;
    EXPECT_FALSE(readAndCheck(makeJson(), si, dag, err));
    EXPECT_NE(err.find("dimension"), std::string::npos) << err;
}

} // namespace
