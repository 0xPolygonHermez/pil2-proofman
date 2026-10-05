#include "pilfflonk_info.hpp"

#include <algorithm>
#include <initializer_list>
#include <limits>

#include <nlohmann/json.hpp>

#include "pilfflonk_error.hpp"
#include "pilfflonk_json.hpp"

namespace PilFflonk {

namespace {

using json = nlohmann::json;

// The largest nBits there is: r - 1 = 2^28 · odd (pilfflonk/docs/protocol.md#notation).
constexpr uint64_t MAX_NBITS = 28;

// The pilfflonkinfo's values, and its errors: "pilfflonkinfo: <where>: <what>".
constexpr JsonReader pilfflonkinfo("pilfflonkinfo");

std::string at(const std::string &where, const char *key) { return where.empty() ? key : where + "." + key; }

std::string at(const std::string &where, size_t index) { return where + "[" + std::to_string(index) + "]"; }

// That `object` is an object with every key of `required`, and no key but those and `optional`.
const json &object(const json &value, const std::string &where, std::initializer_list<const char *> required,
                   std::initializer_list<const char *> optional = {}) {
    if (!value.is_object()) {
        pilfflonkinfo.fail(where, "must be an object");
    }
    for (const char *key : required) {
        if (!value.contains(key)) {
            pilfflonkinfo.fail(where, std::string("has no field \"") + key + "\"");
        }
    }
    for (auto it = value.begin(); it != value.end(); ++it) {
        auto named = [&](const char *key) { return it.key() == key; };
        if (std::none_of(required.begin(), required.end(), named) &&
            std::none_of(optional.begin(), optional.end(), named)) {
            pilfflonkinfo.fail(where, "has an unknown field \"" + it.key() + "\"");
        }
    }
    return value;
}

int64_t i64(const json &value, const std::string &where) {
    if (value.is_number_unsigned()) {
        const uint64_t u = value.get<uint64_t>();
        if (u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            pilfflonkinfo.fail(where, "does not fit in 64 signed bits");
        }
        return static_cast<int64_t>(u);
    }
    if (!value.is_number_integer()) {
        pilfflonkinfo.fail(where, "must be an integer");
    }
    return value.get<int64_t>();
}

template <typename T, typename Read>
std::vector<T> list(const json &value, const std::string &where, Read read) {
    std::vector<T> out;
    const json &items = pilfflonkinfo.array(value, where);
    out.reserve(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        out.push_back(read(items[i], at(where, i)));
    }
    return out;
}

std::vector<uint64_t> lengths(const json &entry, const std::string &where) {
    if (!entry.contains("lengths")) {
        return {};
    }
    return list<uint64_t>(entry["lengths"], at(where, "lengths"), [](const json &length, const std::string &field) {
        return pilfflonkinfo.u64(length, field);
    });
}

PolMapEntry polMapEntry(const json &value, const std::string &where) {
    const json &j = object(value, where, {"stage", "name", "dim", "polsMapId", "stageId", "stagePos"},
                           {"lengths", "imPol", "expId"});
    PolMapEntry p;
    p.stage = pilfflonkinfo.u64(j["stage"], at(where, "stage"));
    p.name = pilfflonkinfo.str(j["name"], at(where, "name"));
    p.dim = pilfflonkinfo.u64(j["dim"], at(where, "dim"));
    p.polsMapId = pilfflonkinfo.u64(j["polsMapId"], at(where, "polsMapId"));
    p.stageId = pilfflonkinfo.u64(j["stageId"], at(where, "stageId"));
    p.lengths = lengths(j, where);
    p.imPol = false;
    if (j.contains("imPol")) {
        if (!j["imPol"].is_boolean()) {
            pilfflonkinfo.fail(at(where, "imPol"), "must be a boolean");
        }
        p.imPol = j["imPol"].get<bool>();
    }
    p.hasExpId = j.contains("expId");
    p.expId = p.hasExpId ? pilfflonkinfo.u64(j["expId"], at(where, "expId")) : 0;
    p.stagePos = pilfflonkinfo.u64(j["stagePos"], at(where, "stagePos"));
    if (p.dim != 1) {
        pilfflonkinfo.fail(at(where, "dim"), "must be 1: BN128 has no extension field");
    }
    return p;
}

ChallengeMapEntry challengeMapEntry(const json &value, const std::string &where) {
    const json &j = object(value, where, {"name", "stage", "dim", "stageId"});
    ChallengeMapEntry c;
    c.name = pilfflonkinfo.str(j["name"], at(where, "name"));
    c.stage = pilfflonkinfo.u64(j["stage"], at(where, "stage"));
    c.dim = pilfflonkinfo.u64(j["dim"], at(where, "dim"));
    c.stageId = pilfflonkinfo.u64(j["stageId"], at(where, "stageId"));
    if (c.dim != 1) {
        pilfflonkinfo.fail(at(where, "dim"), "must be 1: BN128 has no extension field");
    }
    return c;
}

NameStageEntry nameStageEntry(const json &value, const std::string &where) {
    const json &j = object(value, where, {"name", "stage"}, {"lengths"});
    return NameStageEntry{pilfflonkinfo.str(j["name"], at(where, "name")),
                          pilfflonkinfo.u64(j["stage"], at(where, "stage")), lengths(j, where)};
}

EvMapEntry evMapEntry(const json &value, const std::string &where) {
    const json &j = object(value, where, {"type", "id", "prime", "openingPos"});
    EvMapEntry e;
    const std::string type = pilfflonkinfo.str(j["type"], at(where, "type"));
    if (type == "cm") {
        e.type = PolType::Cm;
    } else if (type == "const") {
        e.type = PolType::Const;
    } else {
        pilfflonkinfo.fail(at(where, "type"), "must be \"cm\" or \"const\", not \"" + type + "\"");
    }
    e.id = pilfflonkinfo.u64(j["id"], at(where, "id"));
    e.prime = i64(j["prime"], at(where, "prime"));
    e.openingPos = pilfflonkinfo.u64(j["openingPos"], at(where, "openingPos"));
    return e;
}

Boundary boundary(const json &value, const std::string &where) {
    const json &j = object(value, where, {"name"}, {"offsetMin", "offsetMax"});
    const std::string name = pilfflonkinfo.str(j["name"], at(where, "name"));
    const bool hasOffsets = j.contains("offsetMin") || j.contains("offsetMax");
    Boundary b{BoundaryType::EveryRow, 0, 0};
    if (name == "everyFrame") {
        object(j, where, {"name", "offsetMin", "offsetMax"});
        b.type = BoundaryType::EveryFrame;
        b.offsetMin = pilfflonkinfo.u64(j["offsetMin"], at(where, "offsetMin"));
        b.offsetMax = pilfflonkinfo.u64(j["offsetMax"], at(where, "offsetMax"));
        return b;
    }
    if (name == "everyRow") {
        b.type = BoundaryType::EveryRow;
    } else if (name == "firstRow") {
        b.type = BoundaryType::FirstRow;
    } else if (name == "lastRow") {
        b.type = BoundaryType::LastRow;
    } else {
        pilfflonkinfo.fail(at(where, "name"), "\"" + name + "\" is not a boundary");
    }
    if (hasOffsets) {
        pilfflonkinfo.fail(where, "only everyFrame has offsets");
    }
    return b;
}

LayoutEntry layoutEntry(const json &value, const std::string &where) {
    const json &j = object(value, where, {"stage", "pols", "k", "offsets", "degree"});
    LayoutEntry f;
    f.stage = pilfflonkinfo.u64(j["stage"], at(where, "stage"));
    f.pols = list<LayoutPol>(j["pols"], at(where, "pols"), [](const json &pol, const std::string &polWhere) {
        const json &p = object(pol, polWhere, {"id", "name"});
        return LayoutPol{pilfflonkinfo.u64(p["id"], at(polWhere, "id")),
                         pilfflonkinfo.str(p["name"], at(polWhere, "name"))};
    });
    f.k = pilfflonkinfo.u64(j["k"], at(where, "k"));
    f.offsets = list<int64_t>(j["offsets"], at(where, "offsets"), i64);
    f.degree = pilfflonkinfo.u64(j["degree"], at(where, "degree"));
    if (f.pols.empty() || f.k != f.pols.size()) {
        pilfflonkinfo.fail(at(where, "k"), "must be the number of pols, at least 1");
    }
    if (f.offsets.empty()) {
        pilfflonkinfo.fail(at(where, "offsets"), "must not be empty");
    }
    return f;
}

// The indices the prover follows: each must point at something.
void checkIndices(const PilfflonkInfo &info) {
    if (info.nBits > MAX_NBITS) {
        pilfflonkinfo.fail("nBits", "must be at most " + std::to_string(MAX_NBITS));
    }
    if (info.nStages == 0) {
        pilfflonkinfo.fail("nStages", "must be at least 1");
    }
    if (info.qDim != 1) {
        pilfflonkinfo.fail("qDim", "must be 1: BN128 has no extension field");
    }
    if (info.nConstants != info.constPolsMap.size()) {
        pilfflonkinfo.fail("nConstants", "must be the number of entries of constPolsMap");
    }
    auto checkMap = [&](const std::vector<PolMapEntry> &map, const std::string &where, bool fixed) {
        for (size_t i = 0; i < map.size(); ++i) {
            const PolMapEntry &p = map[i];
            const bool stageOk = fixed ? p.stage == 0 : (p.stage >= 1 && p.stage <= info.qStage());
            if (!stageOk || p.polsMapId != i) {
                pilfflonkinfo.fail(at(where, i),
                                   "must be of a stage of its map and have polsMapId " + std::to_string(i));
            }
            const std::string section = fixed ? "const" : "cm" + std::to_string(p.stage);
            const auto width = info.mapSectionsN.find(section);
            if (width == info.mapSectionsN.end() || p.stagePos >= width->second) {
                pilfflonkinfo.fail(at(at(where, i), "stagePos"), "must be within mapSectionsN." + section);
            }
        }
    };
    checkMap(info.constPolsMap, "constPolsMap", true);
    checkMap(info.cmPolsMap, "cmPolsMap", false);

    for (size_t i = 0; i < info.evMap.size(); ++i) {
        const EvMapEntry &e = info.evMap[i];
        const size_t mapSize = e.type == PolType::Cm ? info.cmPolsMap.size() : info.constPolsMap.size();
        if (e.id >= mapSize) {
            pilfflonkinfo.fail(at(at("evMap", i), "id"), "is not in the pol map");
        }
        if (e.openingPos >= info.openingPoints.size() || info.openingPoints[e.openingPos] != e.prime) {
            pilfflonkinfo.fail(at(at("evMap", i), "openingPos"), "must be the position of prime in openingPoints");
        }
    }

    for (size_t i = 0; i < info.layout.size(); ++i) {
        const LayoutEntry &f = info.layout[i];
        if (f.stage > info.qStage()) {
            pilfflonkinfo.fail(at(at("layout", i), "stage"), "must be at most nStages + 1");
        }
        // The prover takes the fixed f to be the first ones, and the others in the global order
        // (pilfflonk/docs/protocol.md#global-order): the Rust reader refuses any other order too
        // (Layout::check).
        if (i > 0 && f.stage < info.layout[i - 1].stage) {
            pilfflonkinfo.fail(at(at("layout", i), "stage"),
                               "must not be below the stage of the f before it: the layout goes by ascending stage "
                               "(pilfflonk/docs/protocol.md#global-order)");
        }
        const std::vector<PolMapEntry> &map = f.stage == 0 ? info.constPolsMap : info.cmPolsMap;
        for (size_t j = 0; j < f.pols.size(); ++j) {
            if (f.pols[j].id >= map.size() || map[f.pols[j].id].stage != f.stage) {
                pilfflonkinfo.fail(at(at(at("layout", i), "pols"), j), "must be a column of the stage of its f");
            }
        }
    }
}

} // namespace

PilfflonkInfo PilfflonkInfo::parse(const std::string &text) {
    json j;
    try {
        j = json::parse(text);
    } catch (const json::parse_error &e) {
        throw FormatError(std::string("pilfflonkinfo: not valid JSON: ") + e.what());
    }
    object(j, "", {"name", "airgroupId", "airId", "nBits", "nStages", "nConstants", "cmPolsMap", "constPolsMap",
                   "challengesMap", "airValuesMap", "airgroupValuesMap", "mapSectionsN", "openingPoints", "boundaries",
                   "evMap", "qDeg", "qDim", "maxQDegree", "cExpId", "layout"});

    PilfflonkInfo info;
    info.name = pilfflonkinfo.str(j["name"], "name");
    info.airgroupId = pilfflonkinfo.u64(j["airgroupId"], "airgroupId");
    info.airId = pilfflonkinfo.u64(j["airId"], "airId");
    info.nBits = pilfflonkinfo.u64(j["nBits"], "nBits");
    info.nStages = pilfflonkinfo.u64(j["nStages"], "nStages");
    info.nConstants = pilfflonkinfo.u64(j["nConstants"], "nConstants");
    info.cmPolsMap = list<PolMapEntry>(j["cmPolsMap"], "cmPolsMap", polMapEntry);
    info.constPolsMap = list<PolMapEntry>(j["constPolsMap"], "constPolsMap", polMapEntry);
    info.challengesMap = list<ChallengeMapEntry>(j["challengesMap"], "challengesMap", challengeMapEntry);
    info.airValuesMap = list<NameStageEntry>(j["airValuesMap"], "airValuesMap", nameStageEntry);
    info.airgroupValuesMap = list<NameStageEntry>(j["airgroupValuesMap"], "airgroupValuesMap", nameStageEntry);
    const json &sections = j["mapSectionsN"];
    if (!sections.is_object()) {
        pilfflonkinfo.fail("mapSectionsN", "must be an object");
    }
    for (auto it = sections.begin(); it != sections.end(); ++it) {
        info.mapSectionsN[it.key()] = pilfflonkinfo.u64(it.value(), at("mapSectionsN", it.key().c_str()));
    }
    info.openingPoints = list<int64_t>(j["openingPoints"], "openingPoints", i64);
    info.boundaries = list<Boundary>(j["boundaries"], "boundaries", boundary);
    info.evMap = list<EvMapEntry>(j["evMap"], "evMap", evMapEntry);
    info.qDeg = pilfflonkinfo.u64(j["qDeg"], "qDeg");
    info.qDim = pilfflonkinfo.u64(j["qDim"], "qDim");
    info.maxQDegree = pilfflonkinfo.u64(j["maxQDegree"], "maxQDegree");
    info.cExpId = pilfflonkinfo.u64(j["cExpId"], "cExpId");
    info.layout = list<LayoutEntry>(j["layout"], "layout", layoutEntry);
    checkIndices(info);
    return info;
}

PilfflonkInfo PilfflonkInfo::load(const std::string &path) {
    const std::string text = readText(path, "pilfflonkinfo: cannot open " + path, "pilfflonkinfo: cannot read " + path);
    try {
        return parse(text);
    } catch (const FormatError &e) {
        throw FormatError(path + ": " + e.what());
    }
}

} // namespace PilFflonk
