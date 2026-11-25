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

V3 evalTo(const ConstraintDag &dag, const Values &v)
{
    V3 o;
    evalConstraintDag(dag, v.inputs(), o.c);
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
    EXPECT_TRUE(same(evalTo(dag, v), direct(v)));
    v.evals[4] = v.evals[4] + Goldilocks::one();
    EXPECT_TRUE(same(evalTo(dag, v), direct(v)));
}

// NODES reads a public and a proof value, so null for either is refused; a DAG reading neither accepts null.
TEST(ConstraintDag, VerifierRefusesNullInputsItReads)
{
    StarkInfo si = makeInfo();
    ConstraintDag dag;
    std::string err;
    ASSERT_TRUE(readAndCheck(makeJson(), si, dag, err)) << err;
    const GE one[1] = {Goldilocks::one()};
    EXPECT_FALSE(verifierInputsMissing(si, dag, one, one, true));
    EXPECT_TRUE(verifierInputsMissing(si, dag, nullptr, one, true));
    EXPECT_TRUE(verifierInputsMissing(si, dag, one, nullptr, true));

    si.nPublics = 0;
    ConstraintDag noValues;
    ASSERT_TRUE(readAndCheck(makeJson("[[5,0,0,0]]", "[[0,0,\"x\"]]"), si, noValues, err)) << err;
    EXPECT_FALSE(verifierInputsMissing(si, noValues, nullptr, nullptr, true));
    si.nPublics = 1; // a non-vadcop proof binds its publics into the transcript even when the DAG does not read them
    EXPECT_TRUE(verifierInputsMissing(si, noValues, nullptr, nullptr, false));
}

// Every rejection, reader (check = false) or checker: the DAG, the starkinfo edit and a needle of the error.
struct RejectCase
{
    const char *name, *nodes, *roots;
    int version;
    void (*edit)(StarkInfo &);
    bool check;
    const char *needle;
};

void keep(StarkInfo &) {}

const RejectCase REJECTS[] = {
    {"ForwardReference", "[[2,0,1],[5,0,0,0],[5,0,0,-1]]", "[[0,0,\"x\"]]", 1, keep, false, "forward"},
    {"UnknownOpcode", "[[11,0]]", "[[0,0,\"x\"]]", 1, keep, false, "opcode"},
    {"WrongArity", "[[5,0,0,0],[0,0]]", "[[1,0,\"x\"]]", 1, keep, false, "fields"},
    {"NonCanonicalConst", "[[4,\"18446744069414584321\"]]", "[[0,0,\"x\"]]", 1, keep, false, "malformed"},
    {"UnsortedChildren", "[[5,0,0,0],[5,0,0,-1],[2,1,0]]", "[[2,0,\"x\"]]", 1, keep, false, "sorted"},
    {"WrongVersion", NODES, ROOTS, 2, keep, false, "version"},
    {"UnknownColumn", "[[5,0,9,0]]", "[[0,0,\"x\"]]", 1, keep, true, "evMap"}, // no evMap entry for cm id 9
    {"UnknownChallenge", "[[6,2,5]]", "[[0,0,\"x\"]]", 1, keep, true, "challenge"},
    {"MissingStdVc", NODES, ROOTS, 1, [](StarkInfo &si) { si.challengesMap[1].name = "std_other"; }, true, "std_vc"},
    {"NonEveryRowRoot", NODES, "[[13,0,\"c0\"],[17,1,\"c1\"],[21,0,\"c2\"]]", 1, keep, true, "everyRow"},
    {"DuplicateEvEntry", NODES, ROOTS, 1, [](StarkInfo &si) { si.evMap.push_back(si.evMap[0]); }, true, "duplicate"},
    {"DuplicateChallenge", NODES, ROOTS, 1, [](StarkInfo &si) { si.challengesMap.push_back(si.challengesMap[0]); }, true,
     "duplicate"},
    {"PublicOutOfRange", "[[7,1]]", "[[0,0,\"x\"]]", 1, keep, true, "public"}, // nPublics = 1
    // A col node is ext whatever its column's dim (evals at xi are ext), so the ext slot to corrupt is a challenge.
    {"WrongDimensionChallenge", NODES, ROOTS, 1, [](StarkInfo &si) { si.challengesMap[0].dim = 1; }, true, "dimension"},
};

class ConstraintDagRejects : public testing::TestWithParam<RejectCase>
{
};

TEST_P(ConstraintDagRejects, WithReason)
{
    const RejectCase &c = GetParam();
    StarkInfo si = makeInfo();
    c.edit(si);
    ConstraintDag dag;
    std::string err;
    const json j = makeJson(c.nodes, c.roots, c.version);
    EXPECT_FALSE(c.check ? readAndCheck(j, si, dag, err) : readConstraintDag(j, dag, err));
    EXPECT_NE(err.find(c.needle), std::string::npos) << err;
}

INSTANTIATE_TEST_SUITE_P(ConstraintDag, ConstraintDagRejects, testing::ValuesIn(REJECTS),
                         [](const testing::TestParamInfo<RejectCase> &i) { return std::string(i.param.name); });

} // namespace
