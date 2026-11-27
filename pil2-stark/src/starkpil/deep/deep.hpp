#ifndef DEEP_HPP
#define DEEP_HPP

// The DEEP (FRI first-layer) polynomial, derived from the stark info rather than setup bytecode:
//
//   f(x) = SUM_o vf1^e_o / (x - xi_o) * (SUM_{c in o} vf2^c p_c(x) + K_o)      K_o = -SUM_{c in o} vf2^c evals[evalPos]
//
// xi_o = xi w^openingPoints[o] (w of the trace domain), c = the polynomial's first appearance in the eval map
// keyed by (type, commitId if custom else 0, id), e_o = the later openings that have terms; openings without
// terms are skipped. Host-only and header-only: the GPU table builder includes it.
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "stark_info.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

struct DeepTerm
{
    uint32_t evalPos, vf2Exp;
    EvMap::eType type;
    uint64_t id, commitId, stage, stagePos, dim;
};

struct DeepTables
{
    uint64_t xiIdx, vf1Idx, vf2Idx;  // challengesMap indices
    uint32_t nPols;                  // distinct (type, commitId|0, id)
    std::vector<int64_t> openings;   // starkInfo.openingPoints
    std::vector<DeepTerm> terms;     // by opening, evMap order inside
    std::vector<uint64_t> termStart; // openings.size()+1 entries
};

inline uint64_t deepChallengeIndex(const StarkInfo &si, uint64_t stage, uint64_t stageId, const char *name)
{
    for (uint64_t i = 0; i < si.challengesMap.size(); i++)
        if (si.challengesMap[i].stage == stage && si.challengesMap[i].stageId == stageId) return i;
    zklog.error(std::string("buildDeepTables: no challenge ") + name + " (stage " + std::to_string(stage) + ", stageId " +
                std::to_string(stageId) + ")");
    exitProcess();
    return 0;
}

inline DeepTables buildDeepTables(const StarkInfo &si)
{
    DeepTables t;
    t.xiIdx = deepChallengeIndex(si, si.nStages + 2, 0, "xi");
    t.vf1Idx = deepChallengeIndex(si, si.nStages + 3, 0, "vf1");
    t.vf2Idx = deepChallengeIndex(si, si.nStages + 3, 1, "vf2");
    t.openings = si.openingPoints;

    std::vector<std::vector<DeepTerm>> byOpening(si.openingPoints.size());
    std::map<std::tuple<uint64_t, uint64_t, uint64_t>, uint32_t> polIndex;
    for (uint64_t i = 0; i < si.evMap.size(); i++) {
        const EvMap &ev = si.evMap[i];
        if (ev.openingPos >= byOpening.size()) {
            zklog.error("buildDeepTables: evMap entry " + std::to_string(i) + " has openingPos out of range");
            exitProcess();
        }
        const uint32_t vf2Exp =
            polIndex.emplace(StarkInfo::colKey(ev.type, ev.commitId, ev.id), (uint32_t)polIndex.size()).first->second;
        const PolMap &pol = si.evPol(ev);
        byOpening[ev.openingPos].push_back(
            DeepTerm{(uint32_t)i, vf2Exp, ev.type, ev.id, ev.commitId, pol.stage, pol.stagePos, pol.dim});
    }
    t.nPols = (uint32_t)polIndex.size();
    t.termStart.push_back(0);
    for (const auto &opening : byOpening) {
        t.terms.insert(t.terms.end(), opening.begin(), opening.end());
        t.termStart.push_back(t.terms.size());
    }
    return t;
}

// Flat storage indexed as Goldilocks3::Element: a vector of C arrays does not build under nvcc's host STL.
struct DeepF3Vec
{
    std::vector<Goldilocks::Element> d;
    explicit DeepF3Vec(uint64_t n = 0) : d(n * FIELD_EXTENSION) {}
    Goldilocks3::Element &operator[](uint64_t i) { return *(Goldilocks3::Element *)&d[i * FIELD_EXTENSION]; }
    const Goldilocks3::Element &operator[](uint64_t i) const
    {
        return *(const Goldilocks3::Element *)&d[i * FIELD_EXTENSION];
    }
    uint64_t size() const { return d.size() / FIELD_EXTENSION; }
};

// vf2Pow per polynomial; vf1Pow, K and xis per opening.
struct DeepConsts
{
    DeepF3Vec vf2Pow, vf1Pow, K, xis;
};

inline DeepConsts deepConstants(const DeepTables &t, const Goldilocks::Element *challenges,
                                const Goldilocks::Element *evals, uint64_t nBits)
{
    using F3 = Goldilocks3;
    const F3::Element &xi = *(const F3::Element *)&challenges[t.xiIdx * FIELD_EXTENSION];
    const F3::Element &vf1 = *(const F3::Element *)&challenges[t.vf1Idx * FIELD_EXTENSION];
    const F3::Element &vf2 = *(const F3::Element *)&challenges[t.vf2Idx * FIELD_EXTENSION];
    const uint64_t nOpenings = t.openings.size();

    DeepConsts c;
    c.vf2Pow = DeepF3Vec(t.nPols);
    c.vf1Pow = DeepF3Vec(nOpenings);
    c.K = DeepF3Vec(nOpenings);
    c.xis = DeepF3Vec(nOpenings);

    if (t.nPols > 0) F3::one(c.vf2Pow[0]);
    for (uint64_t i = 1; i < t.nPols; i++) F3::mul(c.vf2Pow[i], c.vf2Pow[i - 1], vf2);

    // Walking the openings backwards, e_o is the number of non-empty ones already passed.
    F3::Element p1;
    F3::one(p1);
    for (uint64_t o = nOpenings; o-- > 0;) {
        F3::copy(c.vf1Pow[o], p1);
        if (t.termStart[o + 1] > t.termStart[o]) F3::mul(p1, p1, vf1);
    }

    const Goldilocks::Element w = Goldilocks::w(nBits);
    for (uint64_t o = 0; o < nOpenings; o++) {
        const int64_t op = t.openings[o];
        Goldilocks::Element wo = Goldilocks::exp(w, (uint64_t)(op < 0 ? -op : op));
        if (op < 0) wo = Goldilocks::inv(wo);
        F3::mul(c.xis[o], xi, wo);

        F3::zero(c.K[o]);
        for (uint64_t j = t.termStart[o]; j < t.termStart[o + 1]; j++) {
            F3::Element m;
            F3::mul(m, c.vf2Pow[t.terms[j].vf2Exp], *(const F3::Element *)&evals[t.terms[j].evalPos * FIELD_EXTENSION]);
            F3::sub(c.K[o], c.K[o], m);
        }
    }
    return c;
}

// out[o] = x - xi*w^o, the denominators of the DEEP quotients at the base-field point x (the callers invert).
inline void deepDenominators(const DeepConsts &c, const Goldilocks::Element &x, Goldilocks3::Element *out)
{
    for (uint64_t o = 0; o < c.xis.size(); o++) Goldilocks3::sub(out[o], x, c.xis[o]);
}

// get(j, out[3]) writes p_c of t.terms[j] at the point (dim 1 fills out[0]); invDen[o] = 1 / (x - xis[o]).
template <class Get>
void deepEvalPoint(const DeepTables &t, const DeepConsts &c, const Goldilocks3::Element *invDen, Get &&get,
                   Goldilocks3::Element &out)
{
    using F3 = Goldilocks3;
    F3::zero(out);
    for (uint64_t o = 0; o < t.openings.size(); o++) {
        if (t.termStart[o + 1] == t.termStart[o]) continue;
        F3::Element acc, v, m;
        F3::copy(acc, c.K[o]);
        for (uint64_t j = t.termStart[o]; j < t.termStart[o + 1]; j++) {
            const DeepTerm &term = t.terms[j];
            F3::zero(v);
            get(j, (Goldilocks::Element *)v);
            if (term.dim == 1) F3::mul(m, c.vf2Pow[term.vf2Exp], v[0]);
            else F3::mul(m, c.vf2Pow[term.vf2Exp], v);
            F3::add(acc, acc, m);
        }
        F3::mul(acc, acc, invDen[o]);
        F3::mul(acc, acc, c.vf1Pow[o]);
        F3::add(out, out, acc);
    }
}

// Strided source of every term's values: point r of term j is base[j][r * stride[j]], dim[j] elements.
struct DeepSource
{
    std::vector<const Goldilocks::Element *> base;
    std::vector<uint64_t> stride, dim;

    auto at(uint64_t r) const
    {
        return [this, r](uint64_t j, Goldilocks::Element out[3]) {
            const Goldilocks::Element *p = base[j] + r * stride[j];
            for (uint64_t d = 0; d < dim[j]; d++) out[d] = p[d];
        };
    }
};

// Stage-1 cm values come from cm1, later stages from aux; extended picks the mapOffsets key.
inline DeepSource deepSource(const StarkInfo &si, const DeepTables &t, bool extended, const Goldilocks::Element *cm1,
                             const Goldilocks::Element *aux, const Goldilocks::Element *consts,
                             const Goldilocks::Element *custom)
{
    DeepSource s;
    s.base.resize(t.terms.size());
    s.stride.resize(t.terms.size());
    s.dim.resize(t.terms.size());
    for (uint64_t j = 0; j < t.terms.size(); j++) {
        const DeepTerm &term = t.terms[j];
        s.dim[j] = term.dim;
        if (term.type == EvMap::eType::_const) {
            s.base[j] = &consts[term.stagePos];
            s.stride[j] = si.nConstants;
            continue;
        }
        const std::string section = si.sectionName(term.type, term.stage, term.commitId);
        // Offset first: GCC 13 miscompiles a temporary key inside (c ? p : q)[...] (built in one arm only).
        const uint64_t offset = si.mapOffsets.at(std::make_pair(section, extended));
        const Goldilocks::Element *buf = term.type == EvMap::eType::custom ? custom : term.stage == 1 ? cm1 : aux;
        s.base[j] = buf + offset + term.stagePos;
        s.stride[j] = si.mapSectionsN.at(section);
    }
    return s;
}

#endif
