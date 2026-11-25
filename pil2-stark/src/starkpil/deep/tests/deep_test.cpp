// The DEEP host core against the definition: the challenge lookup, the polynomial numbering and the
// first-layer polynomial at one point.
#include <gtest/gtest.h>

#include <map>
#include <random>
#include <tuple>
#include <vector>

#include "deep.hpp"

namespace
{

using F3 = Goldilocks3;

PolMap challenge(const std::string &name, uint64_t stage, uint64_t stageId)
{
    PolMap p{};
    p.name = name;
    p.stage = stage;
    p.stageId = stageId;
    p.dim = FIELD_EXTENSION;
    return p;
}

PolMap pol(uint64_t stage, uint64_t stagePos, uint64_t dim)
{
    PolMap p{};
    p.stage = stage;
    p.stagePos = stagePos;
    p.dim = dim;
    return p;
}

EvMap ev(EvMap::eType type, uint64_t id, int64_t prime, uint64_t openingPos, uint64_t commitId = 0)
{
    EvMap e{};
    e.type = type;
    e.id = id;
    e.prime = prime;
    e.commitId = commitId;
    e.openingPos = openingPos;
    return e;
}

std::vector<PolMap> challengesNStages2()
{
    return {challenge("std_alpha", 2, 0), challenge("std_gamma", 2, 1), challenge("std_vc", 3, 0),
            challenge("std_xi", 4, 0),    challenge("std_vf1", 5, 0),   challenge("std_vf2", 5, 1)};
}

TEST(Deep, ChallengeIndicesNStages2)
{
    StarkInfo si;
    si.nStages = 2;
    si.challengesMap = challengesNStages2();
    const DeepTables t = buildDeepTables(si);
    EXPECT_EQ(t.xiIdx, 3u);
    EXPECT_EQ(t.vf1Idx, 4u);
    EXPECT_EQ(t.vf2Idx, 5u);
}

TEST(Deep, ChallengeIndicesNStages1)
{
    StarkInfo si;
    si.nStages = 1;
    si.challengesMap = {challenge("std_vc", 2, 0), challenge("std_xi", 3, 0), challenge("std_vf1", 4, 0),
                        challenge("std_vf2", 4, 1)};
    const DeepTables t = buildDeepTables(si);
    EXPECT_EQ(t.xiIdx, 1u);
    EXPECT_EQ(t.vf1Idx, 2u);
    EXPECT_EQ(t.vf2Idx, 3u);
}

TEST(Deep, PolIndexFirstAppearance)
{
    StarkInfo si;
    si.nStages = 2;
    si.challengesMap = challengesNStages2();
    si.cmPolsMap = {pol(1, 0, 1), pol(2, 0, FIELD_EXTENSION)};
    si.constPolsMap = {pol(0, 0, 1)};
    si.openingPoints = {-1, 0};
    si.evMap = {ev(EvMap::eType::cm, 1, -1, 0), ev(EvMap::eType::_const, 0, 0, 1), ev(EvMap::eType::cm, 0, 0, 1),
                ev(EvMap::eType::cm, 1, 0, 1)};
    const DeepTables t = buildDeepTables(si);
    std::vector<uint32_t> exps;
    for (const DeepTerm &term : t.terms) exps.push_back(term.vf2Exp);
    EXPECT_EQ(exps, (std::vector<uint32_t>{0, 1, 2, 0}));
    EXPECT_EQ(t.termStart, (std::vector<uint64_t>{0, 1, 4}));
    EXPECT_EQ(t.nPols, 3u);
}

// A small air over every source with opening 1 left without evaluations, and random field data.
struct Fixture
{
    static constexpr uint64_t nBits = 4;
    StarkInfo si;
    std::vector<Goldilocks::Element> challenges, evals;
    std::map<std::tuple<int, uint64_t, uint64_t>, std::vector<Goldilocks::Element>> values;
    Goldilocks::Element x;

    Fixture()
    {
        std::mt19937_64 rng(7);
        auto rnd = [&] { return Goldilocks::fromU64(rng() % GOLDILOCKS_PRIME); };
        si.nStages = 2;
        si.challengesMap = challengesNStages2();
        si.cmPolsMap = {pol(1, 0, 1), pol(2, 0, FIELD_EXTENSION), pol(1, 1, 1)};
        si.constPolsMap = {pol(0, 0, 1), pol(0, 1, 1)};
        si.customCommitsMap = {{pol(0, 0, 1)}};
        si.openingPoints = {-1, 0, 1};
        si.evMap = {ev(EvMap::eType::cm, 0, 0, 1),     ev(EvMap::eType::cm, 1, -1, 0),
                    ev(EvMap::eType::_const, 1, 0, 1), ev(EvMap::eType::custom, 0, 0, 1, 0),
                    ev(EvMap::eType::cm, 1, 0, 1),     ev(EvMap::eType::_const, 0, -1, 0)};
        for (uint64_t i = 0; i < si.challengesMap.size() * FIELD_EXTENSION; i++) challenges.push_back(rnd());
        for (uint64_t i = 0; i < si.evMap.size() * FIELD_EXTENSION; i++) evals.push_back(rnd());
        for (const EvMap &e : si.evMap) {
            const PolMap &p = si.evPol(e);
            std::vector<Goldilocks::Element> v(FIELD_EXTENSION, Goldilocks::zero());
            for (uint64_t k = 0; k < p.dim; k++) v[k] = rnd();
            values.emplace(key(e.type, e.commitId, e.id), v);
        }
        x = rnd();
    }

    static std::tuple<int, uint64_t, uint64_t> key(EvMap::eType type, uint64_t commitId, uint64_t id)
    {
        return std::make_tuple((int)type, type == EvMap::eType::custom ? commitId : 0, id);
    }

    void native(Goldilocks3::Element &out) const
    {
        const DeepTables t = buildDeepTables(si);
        const DeepConsts c = deepConstants(t, challenges.data(), evals.data(), nBits);
        std::vector<Goldilocks::Element> invDen(t.openings.size() * FIELD_EXTENSION);
        F3::Element *den = (F3::Element *)invDen.data();
        deepDenominators(c, x, den);
        for (uint64_t o = 0; o < t.openings.size(); o++) F3::inv(den[o], den[o]);
        deepEvalPoint(t, c, (const F3::Element *)invDen.data(),
                      [&](uint64_t j, Goldilocks::Element *v) {
                          const DeepTerm &term = t.terms[j];
                          const auto &val = values.at(key(term.type, term.commitId, term.id));
                          for (uint64_t k = 0; k < term.dim; k++) v[k] = val[k];
                      },
                      out);
    }

    // The definition: per-term inversion of x - xi w^o, Horner on vf1 over the openings with terms.
    void reference(Goldilocks3::Element &out) const
    {
        const F3::Element &xi = *(const F3::Element *)&challenges[3 * FIELD_EXTENSION];
        const F3::Element &vf1 = *(const F3::Element *)&challenges[4 * FIELD_EXTENSION];
        const F3::Element &vf2 = *(const F3::Element *)&challenges[5 * FIELD_EXTENSION];
        std::map<std::tuple<int, uint64_t, uint64_t>, uint64_t> index;
        for (const EvMap &e : si.evMap) index.emplace(key(e.type, e.commitId, e.id), index.size());
        F3::zero(out);
        for (uint64_t o = 0; o < si.openingPoints.size(); o++) {
            const int64_t op = si.openingPoints[o];
            Goldilocks::Element w = Goldilocks::one();
            for (int64_t k = 0; k < (op < 0 ? -op : op); k++) w = w * Goldilocks::w(nBits);
            if (op < 0) w = Goldilocks::inv(w);
            F3::Element xio;
            F3::mul(xio, xi, w);
            F3::Element termO;
            F3::zero(termO);
            bool any = false;
            for (uint64_t i = 0; i < si.evMap.size(); i++) {
                const EvMap &e = si.evMap[i];
                if (e.openingPos != o) continue;
                any = true;
                const auto &val = values.at(key(e.type, e.commitId, e.id));
                F3::Element p = {val[0], val[1], val[2]}, d, den, inv, vp;
                F3::sub(d, p, *(const F3::Element *)&evals[i * FIELD_EXTENSION]);
                F3::one(vp);
                for (uint64_t k = 0; k < index.at(key(e.type, e.commitId, e.id)); k++) F3::mul(vp, vp, vf2);
                F3::mul(d, d, vp);
                F3::sub(den, x, xio);
                F3::inv(inv, den);
                F3::mul(d, d, inv);
                F3::add(termO, termO, d);
            }
            if (!any) continue;
            F3::mul(out, out, vf1);
            F3::add(out, out, termO);
        }
    }
};

bool equal(const Goldilocks3::Element &a, const Goldilocks3::Element &b)
{
    for (uint64_t k = 0; k < FIELD_EXTENSION; k++)
        if (Goldilocks::toU64(a[k]) != Goldilocks::toU64(b[k])) return false;
    return true;
}

TEST(Deep, MatchesDefinition)
{
    const Fixture f;
    F3::Element got, want;
    f.native(got);
    f.reference(want);
    EXPECT_TRUE(equal(got, want));
}

// One point laid out as the prover sees it (row R of the extended row-major buffers) and as the verifier sees it
// (query Q of the per-query opened rows); both accessors must read the same values.
struct Layouts
{
    static constexpr uint64_t NExt = 8, R = 5, nQueries = 3, Q = 1;
    Fixture f;
    std::vector<Goldilocks::Element> pAux, pConst, pCustom; // prover
    std::vector<Goldilocks::Element> vTrace, vAux, vConst, vCustom; // verifier

    Layouts()
    {
        StarkInfo &si = f.si;
        si.nConstants = 2;
        si.customCommits = {CustomCommits{}};
        si.customCommits[0].name = "rom";
        // Prover: cm1 and cm2 extended live in aux_trace; verifier: cm1 in trace, cm2 in aux_trace.
        si.mapSectionsN = {{"cm1", 2}, {"cm2", FIELD_EXTENSION}, {"rom0", 1}};
        si.mapOffsets = {{{"cm1", true}, 0},   {{"cm2", true}, NExt * 2}, {{"rom0", true}, 0},
                         {{"cm1", false}, 0},  {{"cm2", false}, 4},       {{"rom0", false}, 0}};
        std::mt19937_64 rng(11);
        auto fill = [&](std::vector<Goldilocks::Element> &v, uint64_t n) {
            v.resize(n);
            for (auto &e : v) e = Goldilocks::fromU64(rng() % GOLDILOCKS_PRIME);
        };
        fill(pAux, NExt * (2 + FIELD_EXTENSION));
        fill(pConst, NExt * 2);
        fill(pCustom, NExt);
        fill(vTrace, nQueries * 2);
        fill(vAux, 4 + nQueries * FIELD_EXTENSION);
        fill(vConst, nQueries * 2);
        fill(vCustom, nQueries);
        for (const EvMap &e : si.evMap) {
            const auto &val = f.values.at(Fixture::key(e.type, e.commitId, e.id));
            const PolMap &p = e.type == EvMap::eType::cm       ? si.cmPolsMap[e.id]
                              : e.type == EvMap::eType::custom ? si.customCommitsMap[e.commitId][e.id]
                                                               : si.constPolsMap[e.id];
            for (uint64_t k = 0; k < p.dim; k++) {
                if (e.type == EvMap::eType::cm) {
                    const std::string sec = "cm" + std::to_string(p.stage);
                    const uint64_t n = si.mapSectionsN.at(sec);
                    pAux[si.mapOffsets.at({sec, true}) + R * n + p.stagePos + k] = val[k];
                    (p.stage == 1 ? vTrace : vAux)[si.mapOffsets.at({sec, false}) + Q * n + p.stagePos + k] = val[k];
                } else if (e.type == EvMap::eType::custom) {
                    pCustom[R + p.stagePos + k] = val[k];
                    vCustom[Q + p.stagePos + k] = val[k];
                } else {
                    pConst[R * 2 + p.stagePos + k] = val[k];
                    vConst[Q * 2 + p.stagePos + k] = val[k];
                }
            }
        }
    }

    // The native value at f.x through a strided source at point r.
    void eval(const DeepSource &src, uint64_t r, Goldilocks3::Element &out) const
    {
        const DeepTables t = buildDeepTables(f.si);
        const DeepConsts c = deepConstants(t, f.challenges.data(), f.evals.data(), Fixture::nBits);
        std::vector<Goldilocks::Element> invDen(t.openings.size() * FIELD_EXTENSION);
        F3::Element *den = (F3::Element *)invDen.data();
        deepDenominators(c, f.x, den);
        for (uint64_t o = 0; o < t.openings.size(); o++) F3::inv(den[o], den[o]);
        deepEvalPoint(t, c, (const F3::Element *)invDen.data(), src.at(r), out);
    }

    DeepSource prover() const
    {
        return deepSource(f.si, buildDeepTables(f.si), true, pAux.data(), pAux.data(), pConst.data(), pCustom.data());
    }
    DeepSource verifier() const
    {
        return deepSource(f.si, buildDeepTables(f.si), false, vTrace.data(), vAux.data(), vConst.data(), vCustom.data());
    }
};

TEST(Deep, VerifierAccessorMatchesProverAccessor)
{
    const Layouts l;
    F3::Element p, v, want;
    l.eval(l.prover(), Layouts::R, p);
    l.eval(l.verifier(), Layouts::Q, v);
    l.f.native(want);
    EXPECT_TRUE(equal(p, want));
    EXPECT_TRUE(equal(v, want));
}

// The verifier's native DEEP no longer matches the committed f_0 once any one opened value is perturbed.
TEST(Deep, VerifierRejectsPerturbedOpening)
{
    Layouts l;
    F3::Element honest;
    l.eval(l.verifier(), Layouts::Q, honest);
    // Every cell of query Q some term reads: cm1 col 0 (col 1 is never opened); cm2 cols 0..2; consts 0,1; custom 0.
    const std::vector<std::pair<std::vector<Goldilocks::Element> *, uint64_t>> opened = {
        {&l.vTrace, Layouts::Q * 2},       {&l.vAux, 4 + Layouts::Q * 3},  {&l.vAux, 4 + Layouts::Q * 3 + 1},
        {&l.vAux, 4 + Layouts::Q * 3 + 2}, {&l.vConst, Layouts::Q * 2},    {&l.vConst, Layouts::Q * 2 + 1},
        {&l.vCustom, Layouts::Q}};
    for (const auto &[buf, i] : opened) {
        const Goldilocks::Element saved = (*buf)[i];
        (*buf)[i] = saved + Goldilocks::one();
        F3::Element got;
        l.eval(l.verifier(), Layouts::Q, got);
        (*buf)[i] = saved;
        EXPECT_FALSE(equal(got, honest)) << "cell " << i;
    }
    // Cells of another query, or of an unopened column, are not read for Q.
    l.vTrace[0] = l.vTrace[0] + Goldilocks::one();
    l.vTrace[Layouts::Q * 2 + 1] = l.vTrace[Layouts::Q * 2 + 1] + Goldilocks::one();
    F3::Element other;
    l.eval(l.verifier(), Layouts::Q, other);
    EXPECT_TRUE(equal(other, honest));
}

} // namespace
