#ifndef CONSTRAINT_DAG_HPP
#define CONSTRAINT_DAG_HPP

// The constraint DAG of verifierinfo.json (key constraintDag) and its evaluation at xi.
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "stark_info.hpp"
#include "zkassert.hpp"

enum class DagKind : uint8_t { Add, Sub, Mul, Neg, Const, Col, Challenge, Public, ProofValue, AirValue, AirgroupValue };

struct DagNode
{
    DagKind kind;
    EvMap::eType colType;
    uint64_t a, b, commitId, id;
    int64_t offset;
    Goldilocks::Element value;
};

// Where a leaf's value lives in DagInputs: element offset and dimension.
enum class DagSrc : uint8_t { Number, Eval, Challenge, Public, ProofValue, AirValue, AirgroupValue };
struct DagLeaf
{
    DagSrc src;
    uint64_t pos, dim;
};

struct ConstraintDag
{
    std::vector<DagNode> nodes;
    std::vector<uint64_t> rootNode;
    std::vector<uint8_t> rootBoundary; // labels ignored
    // Filled by checkConstraintDag: each node's resolved leaf (unused for an op node) and std_vc's index.
    std::vector<DagLeaf> leaves;
    uint64_t vc = 0;
};

namespace constraint_dag_detail
{
using Src = DagSrc;
using Leaf = DagLeaf;

inline bool fail(std::string &err, const std::string &msg)
{
    err = "constraintDag: " + msg;
    return false;
}

inline bool readU64(const json &j, uint64_t &out)
{
    if (!j.is_number_unsigned()) return false;
    out = j.get<uint64_t>();
    return true;
}

// Canonical decimal: digits only, no leading zero, < p.
inline bool readConst(const json &j, Goldilocks::Element &out)
{
    if (!j.is_string()) return false;
    const std::string s = j.get<std::string>();
    if (s.empty() || s.size() > 20 || (s.size() > 1 && s[0] == '0')) return false;
    unsigned __int128 v = 0;
    for (char c : s)
    {
        if (c < '0' || c > '9') return false;
        v = v * 10 + uint64_t(c - '0');
    }
    if (v >= GOLDILOCKS_PRIME) return false;
    out = Goldilocks::fromU64(uint64_t(v));
    return true;
}

// The only resolver of col/challenge/value references (mirrors Rust LowerCtx::resolve and its unique-key rule).
struct Resolver
{
    std::map<std::tuple<uint64_t, uint64_t, uint64_t, int64_t>, uint64_t> ev;
    std::map<std::pair<uint64_t, uint64_t>, uint64_t> challenges;
    uint64_t vc = 0, nPublics = 0;
    struct Value { uint64_t pos, dim; };
    std::vector<Value> proofValues, airValues, airgroupValues;

    static std::tuple<uint64_t, uint64_t, uint64_t, int64_t> evKey(EvMap::eType t, uint64_t commitId, uint64_t id, int64_t prime)
    {
        return {uint64_t(t), t == EvMap::custom ? commitId : 0, id, prime};
    }

    // Values are packed consecutively, StarkInfo::valueDim elements each.
    static void pack(const std::vector<PolMap> &map, std::vector<Value> &out)
    {
        uint64_t pos = 0;
        for (const PolMap &m : map)
        {
            out.push_back({pos, StarkInfo::valueDim(m)});
            pos += StarkInfo::valueDim(m);
        }
    }

    bool init(const StarkInfo &si, std::string &err)
    {
        for (uint64_t k = 0; k < si.evMap.size(); k++)
        {
            const EvMap &e = si.evMap[k];
            if (!ev.emplace(evKey(e.type, e.commitId, e.id, e.prime), k).second)
                return fail(err, "duplicate evMap entry " + std::to_string(k));
        }
        bool hasVc = false;
        for (uint64_t k = 0; k < si.challengesMap.size(); k++)
        {
            const PolMap &c = si.challengesMap[k];
            if (!challenges.emplace(std::make_pair(c.stage, c.stageId), k).second)
                return fail(err, "duplicate challenge " + std::to_string(c.stage) + "." + std::to_string(c.stageId));
            if (c.dim != FIELD_EXTENSION) return fail(err, "challenge " + c.name + " has dimension " + std::to_string(c.dim));
            if (c.name == "std_vc")
            {
                if (hasVc) return fail(err, "duplicate std_vc");
                hasVc = true;
                vc = k;
            }
        }
        if (!hasVc) return fail(err, "no std_vc challenge");
        nPublics = si.nPublics;
        pack(si.proofValuesMap, proofValues);
        pack(si.airValuesMap, airValues);
        pack(si.airgroupValuesMap, airgroupValues);
        return true;
    }

    static bool value(const std::vector<Value> &vals, uint64_t id, Src src, Leaf &out, const char *what, std::string &err)
    {
        if (id >= vals.size()) return fail(err, std::string("unknown ") + what + " " + std::to_string(id));
        out = {src, vals[id].pos, vals[id].dim};
        return true;
    }

    // Leaf node -> its slot; false for an unresolved reference (an operation node is not a leaf).
    bool resolve(const DagNode &n, Leaf &out, std::string &err) const
    {
        switch (n.kind)
        {
        case DagKind::Const: out = {Src::Number, 0, 1}; return true;
        case DagKind::Col:
        {
            auto it = ev.find(evKey(n.colType, n.commitId, n.id, n.offset));
            if (it == ev.end())
                return fail(err, "no evMap entry for column type " + std::to_string(n.colType) + " id " + std::to_string(n.id) +
                                     " offset " + std::to_string(n.offset));
            out = {Src::Eval, it->second * FIELD_EXTENSION, FIELD_EXTENSION};
            return true;
        }
        case DagKind::Challenge:
        {
            auto it = challenges.find({n.a, n.b});
            if (it == challenges.end()) return fail(err, "unknown challenge " + std::to_string(n.a) + "." + std::to_string(n.b));
            out = {Src::Challenge, it->second * FIELD_EXTENSION, FIELD_EXTENSION};
            return true;
        }
        case DagKind::Public:
            if (n.id >= nPublics) return fail(err, "public " + std::to_string(n.id) + " out of range");
            out = {Src::Public, n.id, 1};
            return true;
        case DagKind::ProofValue: return value(proofValues, n.id, Src::ProofValue, out, "proof value", err);
        case DagKind::AirValue: return value(airValues, n.id, Src::AirValue, out, "air value", err);
        case DagKind::AirgroupValue: return value(airgroupValues, n.id, Src::AirgroupValue, out, "airgroup value", err);
        default: return fail(err, "not a leaf");
        }
    }
};

inline bool isOp(DagKind k) { return k == DagKind::Add || k == DagKind::Sub || k == DagKind::Mul || k == DagKind::Neg; }

} // namespace constraint_dag_detail

// Format checks: version, forward refs, unknown opcodes/arity, const < p, sorted add/mul children.
inline bool readConstraintDag(const json &j, ConstraintDag &out, std::string &err)
{
    using namespace constraint_dag_detail;
    out = ConstraintDag{};
    if (!j.is_object()) return fail(err, "not an object");
    if (!j.contains("version") || !j["version"].is_number_unsigned() || j["version"].get<uint64_t>() != 1)
        return fail(err, "unsupported version (expected 1)");
    if (!j.contains("field") || j["field"] != "gl" || !j.contains("extDim") || j["extDim"] != FIELD_EXTENSION)
        return fail(err, "field must be gl with extDim 3");
    if (!j.contains("nodes") || !j["nodes"].is_array() || !j.contains("roots") || !j["roots"].is_array())
        return fail(err, "nodes and roots must be arrays");

    const json &nodes = j["nodes"];
    out.nodes.reserve(nodes.size());
    for (uint64_t i = 0; i < nodes.size(); i++)
    {
        const json &a = nodes[i];
        const std::string at = "node " + std::to_string(i) + ": ";
        uint64_t op;
        if (!a.is_array() || a.empty() || !readU64(a[0], op)) return fail(err, at + "not [opcode, ...]");
        static const size_t arity[] = {3, 3, 3, 2, 2, 4, 3, 2, 2, 2, 2}; // col: 4, or 5 for custom
        if (op > uint64_t(DagKind::AirgroupValue)) return fail(err, at + "unknown opcode " + std::to_string(op));
        DagNode n{};
        n.kind = DagKind(op);
        size_t want = arity[op];
        if (n.kind == DagKind::Col && a.size() >= 2 && a[1] == uint64_t(EvMap::custom)) want = 5;
        if (a.size() != want) return fail(err, at + "wrong number of fields for opcode " + std::to_string(op));
        bool ok = true;
        switch (n.kind)
        {
        case DagKind::Add:
        case DagKind::Sub:
        case DagKind::Mul: ok = readU64(a[1], n.a) && readU64(a[2], n.b); break;
        case DagKind::Neg: ok = readU64(a[1], n.a); break;
        case DagKind::Const: ok = readConst(a[1], n.value); break;
        case DagKind::Col:
        {
            uint64_t t;
            ok = readU64(a[1], t) && t <= uint64_t(EvMap::custom) && readU64(a[2], n.id) && a[3].is_number_integer() &&
                 (want == 4 || readU64(a[4], n.commitId));
            if (ok)
            {
                n.colType = EvMap::eType(t);
                n.offset = a[3].get<int64_t>();
            }
            break;
        }
        case DagKind::Challenge: ok = readU64(a[1], n.a) && readU64(a[2], n.b); break;
        default: ok = readU64(a[1], n.id); break;
        }
        if (!ok) return fail(err, at + "malformed field");
        if (isOp(n.kind))
        {
            if (n.a >= i || (n.kind != DagKind::Neg && n.b >= i)) return fail(err, at + "forward or self reference");
            if ((n.kind == DagKind::Add || n.kind == DagKind::Mul) && n.a > n.b) return fail(err, at + "add/mul children not sorted");
        }
        out.nodes.push_back(n);
    }

    const json &roots = j["roots"];
    if (roots.empty()) return fail(err, "no roots");
    for (uint64_t r = 0; r < roots.size(); r++)
    {
        uint64_t node, boundary;
        const json &a = roots[r];
        if (!a.is_array() || a.size() != 3 || !readU64(a[0], node) || !readU64(a[1], boundary) || boundary > 255 || !a[2].is_string())
            return fail(err, "root " + std::to_string(r) + ": not [node, boundary, label]");
        if (node >= out.nodes.size()) return fail(err, "root " + std::to_string(r) + ": node out of range (forward reference)");
        out.rootNode.push_back(node);
        out.rootBoundary.push_back(uint8_t(boundary));
    }
    return true;
}

// Resolves col/challenge/value refs and std_vc once, into dag.leaves/dag.vc; every root everyRow (0); dimensions;
// unique evMap/challenge matches.
inline bool checkConstraintDag(ConstraintDag &dag, const StarkInfo &si, std::string &err)
{
    using namespace constraint_dag_detail;
    dag.leaves.clear();
    Resolver res;
    if (!res.init(si, err)) return false;
    std::vector<Leaf> leaves(dag.nodes.size(), Leaf{Src::Number, 0, 0});
    for (uint64_t i = 0; i < dag.nodes.size(); i++)
    {
        if (!isOp(dag.nodes[i].kind) && !res.resolve(dag.nodes[i], leaves[i], err))
        {
            err += " (node " + std::to_string(i) + ")";
            return false;
        }
    }
    for (uint64_t r = 0; r < dag.rootNode.size(); r++)
        if (dag.rootBoundary[r] != 0) return fail(err, "root " + std::to_string(r) + " is not everyRow");
    dag.leaves = std::move(leaves);
    dag.vc = res.vc;
    return true;
}

// evals/challenges: 3 elements per evMap/challengesMap entry; publics: 1 per entry; proofValues/airValues/airgroupValues:
// consecutive, each map entry taking 1 element if its stage is 1, else 3.
struct DagInputs
{
    const Goldilocks::Element *evals, *challenges, *publics, *proofValues, *airValues, *airgroupValues;
    Goldilocks::Element zi[3];
};

// out = Horner(roots, vc) * zi; the DAG must have passed checkConstraintDag (its resolved leaves).
inline void evalConstraintDag(const ConstraintDag &dag, const DagInputs &in, Goldilocks::Element out[3])
{
    using namespace constraint_dag_detail;
    using E3 = Goldilocks3::Element;
    zkassertpermanent(dag.leaves.size() == dag.nodes.size());
    std::vector<Goldilocks::Element> v(dag.nodes.size() * FIELD_EXTENSION, Goldilocks::zero());
    auto at = [&](uint64_t k) -> E3 & { return *(E3 *)&v[k * FIELD_EXTENSION]; };
    for (uint64_t i = 0; i < dag.nodes.size(); i++)
    {
        const DagNode &n = dag.nodes[i];
        E3 &d = at(i);
        switch (n.kind)
        {
        case DagKind::Add: Goldilocks3::add(d, at(n.a), at(n.b)); continue;
        case DagKind::Sub: Goldilocks3::sub(d, at(n.a), at(n.b)); continue;
        case DagKind::Mul: Goldilocks3::mul(d, at(n.a), at(n.b)); continue;
        case DagKind::Neg:
        {
            E3 zero = {Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero()};
            Goldilocks3::sub(d, zero, at(n.a));
            continue;
        }
        default: break;
        }
        const Leaf &leaf = dag.leaves[i];
        const Goldilocks::Element *srcs[] = {&n.value, in.evals, in.challenges, in.publics, in.proofValues, in.airValues, in.airgroupValues};
        const Goldilocks::Element *p = srcs[uint8_t(leaf.src)] + leaf.pos;
        for (uint64_t k = 0; k < leaf.dim; k++) d[k] = p[k]; // a base value embeds with zero high limbs
    }

    // The protocol fold: acc = c_0, acc = vc*acc + c_i, then times zi.
    const E3 &vc = *(const E3 *)&in.challenges[dag.vc * FIELD_EXTENSION];
    E3 acc;
    for (uint64_t r = 0; r < dag.rootNode.size(); r++)
    {
        if (r == 0)
        {
            for (int k = 0; k < 3; k++) acc[k] = at(dag.rootNode[0])[k];
            continue;
        }
        Goldilocks3::mul(acc, vc, acc);
        Goldilocks3::add(acc, acc, at(dag.rootNode[r]));
    }
    Goldilocks3::mul(*(E3 *)out, acc, *(const E3 *)in.zi);
}

#endif
