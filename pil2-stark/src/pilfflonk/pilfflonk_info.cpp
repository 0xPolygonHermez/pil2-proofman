#include "pilfflonk_info.hpp"

#include <algorithm>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>

#include <nlohmann/json.hpp>

#include "pilfflonk_error.hpp"

namespace PilFflonk {

namespace {

using json = nlohmann::json;

// The largest nBits there is: r - 1 = 2^28 · odd (pilfflonk/docs/protocol.md#notation).
constexpr uint64_t MAX_NBITS = 28;

// `where` is the field, as a path from the top-level object ("" for that object itself).
[[noreturn]] void fail(const std::string &where, const std::string &what) {
    throw FormatError("pilfflonkinfo: " + (where.empty() ? std::string("the file") : where) + ": " + what);
}

std::string at(const std::string &where, const char *key) { return where.empty() ? key : where + "." + key; }

std::string at(const std::string &where, size_t index) { return where + "[" + std::to_string(index) + "]"; }

// That `object` is an object with every key of `required`, and no key but those and `optional`.
const json &object(const json &value, const std::string &where, std::initializer_list<const char *> required,
                   std::initializer_list<const char *> optional = {}) {
    if (!value.is_object()) {
        fail(where, "must be an object");
    }
    for (const char *key : required) {
        if (!value.contains(key)) {
            fail(where, std::string("has no field \"") + key + "\"");
        }
    }
    for (auto it = value.begin(); it != value.end(); ++it) {
        auto named = [&](const char *key) { return it.key() == key; };
        if (std::none_of(required.begin(), required.end(), named) &&
            std::none_of(optional.begin(), optional.end(), named)) {
            fail(where, "has an unknown field \"" + it.key() + "\"");
        }
    }
    return value;
}

uint64_t u64(const json &value, const std::string &where) {
    if (!value.is_number_unsigned()) {
        fail(where, "must be an unsigned integer");
    }
    return value.get<uint64_t>();
}

int64_t i64(const json &value, const std::string &where) {
    if (value.is_number_unsigned()) {
        const uint64_t u = value.get<uint64_t>();
        if (u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            fail(where, "does not fit in 64 signed bits");
        }
        return static_cast<int64_t>(u);
    }
    if (!value.is_number_integer()) {
        fail(where, "must be an integer");
    }
    return value.get<int64_t>();
}

std::string str(const json &value, const std::string &where) {
    if (!value.is_string()) {
        fail(where, "must be a string");
    }
    return value.get<std::string>();
}

const json &array(const json &value, const std::string &where) {
    if (!value.is_array()) {
        fail(where, "must be an array");
    }
    return value;
}

template <typename T, typename Read>
std::vector<T> list(const json &value, const std::string &where, Read read) {
    std::vector<T> out;
    const json &items = array(value, where);
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
    return list<uint64_t>(entry["lengths"], at(where, "lengths"), u64);
}

PolMapEntry polMapEntry(const json &value, const std::string &where) {
    const json &j = object(value, where, {"stage", "name", "dim", "polsMapId", "stageId", "stagePos"},
                           {"lengths", "imPol", "expId"});
    PolMapEntry p;
    p.stage = u64(j["stage"], at(where, "stage"));
    p.name = str(j["name"], at(where, "name"));
    p.dim = u64(j["dim"], at(where, "dim"));
    p.polsMapId = u64(j["polsMapId"], at(where, "polsMapId"));
    p.stageId = u64(j["stageId"], at(where, "stageId"));
    p.lengths = lengths(j, where);
    p.imPol = false;
    if (j.contains("imPol")) {
        if (!j["imPol"].is_boolean()) {
            fail(at(where, "imPol"), "must be a boolean");
        }
        p.imPol = j["imPol"].get<bool>();
    }
    p.hasExpId = j.contains("expId");
    p.expId = p.hasExpId ? u64(j["expId"], at(where, "expId")) : 0;
    p.stagePos = u64(j["stagePos"], at(where, "stagePos"));
    if (p.dim != 1) {
        fail(at(where, "dim"), "must be 1: BN128 has no extension field");
    }
    return p;
}

ChallengeMapEntry challengeMapEntry(const json &value, const std::string &where) {
    const json &j = object(value, where, {"name", "stage", "dim", "stageId"});
    ChallengeMapEntry c;
    c.name = str(j["name"], at(where, "name"));
    c.stage = u64(j["stage"], at(where, "stage"));
    c.dim = u64(j["dim"], at(where, "dim"));
    c.stageId = u64(j["stageId"], at(where, "stageId"));
    if (c.dim != 1) {
        fail(at(where, "dim"), "must be 1: BN128 has no extension field");
    }
    return c;
}

NameStageEntry nameStageEntry(const json &value, const std::string &where) {
    const json &j = object(value, where, {"name", "stage"}, {"lengths"});
    return NameStageEntry{str(j["name"], at(where, "name")), u64(j["stage"], at(where, "stage")), lengths(j, where)};
}

EvMapEntry evMapEntry(const json &value, const std::string &where) {
    const json &j = object(value, where, {"type", "id", "prime", "openingPos"});
    EvMapEntry e;
    const std::string type = str(j["type"], at(where, "type"));
    if (type == "cm") {
        e.type = PolType::Cm;
    } else if (type == "const") {
        e.type = PolType::Const;
    } else {
        fail(at(where, "type"), "must be \"cm\" or \"const\", not \"" + type + "\"");
    }
    e.id = u64(j["id"], at(where, "id"));
    e.prime = i64(j["prime"], at(where, "prime"));
    e.openingPos = u64(j["openingPos"], at(where, "openingPos"));
    return e;
}

Boundary boundary(const json &value, const std::string &where) {
    const json &j = object(value, where, {"name"}, {"offsetMin", "offsetMax"});
    const std::string name = str(j["name"], at(where, "name"));
    const bool hasOffsets = j.contains("offsetMin") || j.contains("offsetMax");
    Boundary b{BoundaryType::EveryRow, 0, 0};
    if (name == "everyFrame") {
        object(j, where, {"name", "offsetMin", "offsetMax"});
        b.type = BoundaryType::EveryFrame;
        b.offsetMin = u64(j["offsetMin"], at(where, "offsetMin"));
        b.offsetMax = u64(j["offsetMax"], at(where, "offsetMax"));
        return b;
    }
    if (name == "everyRow") {
        b.type = BoundaryType::EveryRow;
    } else if (name == "firstRow") {
        b.type = BoundaryType::FirstRow;
    } else if (name == "lastRow") {
        b.type = BoundaryType::LastRow;
    } else {
        fail(at(where, "name"), "\"" + name + "\" is not a boundary");
    }
    if (hasOffsets) {
        fail(where, "only everyFrame has offsets");
    }
    return b;
}

LayoutEntry layoutEntry(const json &value, const std::string &where) {
    const json &j = object(value, where, {"stage", "pols", "k", "offsets", "degree"});
    LayoutEntry f;
    f.stage = u64(j["stage"], at(where, "stage"));
    f.pols = list<LayoutPol>(j["pols"], at(where, "pols"), [](const json &pol, const std::string &polWhere) {
        const json &p = object(pol, polWhere, {"id", "name"});
        return LayoutPol{u64(p["id"], at(polWhere, "id")), str(p["name"], at(polWhere, "name"))};
    });
    f.k = u64(j["k"], at(where, "k"));
    f.offsets = list<int64_t>(j["offsets"], at(where, "offsets"), i64);
    f.degree = u64(j["degree"], at(where, "degree"));
    if (f.pols.empty() || f.k != f.pols.size()) {
        fail(at(where, "k"), "must be the number of pols, at least 1");
    }
    if (f.offsets.empty()) {
        fail(at(where, "offsets"), "must not be empty");
    }
    return f;
}

// The indices the prover follows: each must point at something.
void checkIndices(const PilfflonkInfo &info) {
    if (info.nBits > MAX_NBITS) {
        fail("nBits", "must be at most " + std::to_string(MAX_NBITS));
    }
    if (info.nStages == 0) {
        fail("nStages", "must be at least 1");
    }
    if (info.qDim != 1) {
        fail("qDim", "must be 1: BN128 has no extension field");
    }
    if (info.nConstants != info.constPolsMap.size()) {
        fail("nConstants", "must be the number of entries of constPolsMap");
    }
    auto checkMap = [&](const std::vector<PolMapEntry> &map, const std::string &where, bool fixed) {
        for (size_t i = 0; i < map.size(); ++i) {
            const PolMapEntry &p = map[i];
            const bool stageOk = fixed ? p.stage == 0 : (p.stage >= 1 && p.stage <= info.qStage());
            if (!stageOk || p.polsMapId != i) {
                fail(at(where, i), "must be of a stage of its map and have polsMapId " + std::to_string(i));
            }
            const std::string section = fixed ? "const" : "cm" + std::to_string(p.stage);
            const auto width = info.mapSectionsN.find(section);
            if (width == info.mapSectionsN.end() || p.stagePos >= width->second) {
                fail(at(at(where, i), "stagePos"), "must be within mapSectionsN." + section);
            }
        }
    };
    checkMap(info.constPolsMap, "constPolsMap", true);
    checkMap(info.cmPolsMap, "cmPolsMap", false);

    for (size_t i = 0; i < info.evMap.size(); ++i) {
        const EvMapEntry &e = info.evMap[i];
        const size_t mapSize = e.type == PolType::Cm ? info.cmPolsMap.size() : info.constPolsMap.size();
        if (e.id >= mapSize) {
            fail(at(at("evMap", i), "id"), "is not in the pol map");
        }
        if (e.openingPos >= info.openingPoints.size() || info.openingPoints[e.openingPos] != e.prime) {
            fail(at(at("evMap", i), "openingPos"), "must be the position of prime in openingPoints");
        }
    }

    for (size_t i = 0; i < info.layout.size(); ++i) {
        const LayoutEntry &f = info.layout[i];
        if (f.stage > info.qStage()) {
            fail(at(at("layout", i), "stage"), "must be at most nStages + 1");
        }
        // The prover takes the fixed f to be the first ones, and the others in the global order
        // (pilfflonk/docs/protocol.md#global-order): the Rust reader refuses any other order too
        // (Layout::check).
        if (i > 0 && f.stage < info.layout[i - 1].stage) {
            fail(at(at("layout", i), "stage"),
                 "must not be below the stage of the f before it: the layout goes by ascending stage "
                 "(pilfflonk/docs/protocol.md#global-order)");
        }
        const std::vector<PolMapEntry> &map = f.stage == 0 ? info.constPolsMap : info.cmPolsMap;
        for (size_t j = 0; j < f.pols.size(); ++j) {
            if (f.pols[j].id >= map.size() || map[f.pols[j].id].stage != f.stage) {
                fail(at(at(at("layout", i), "pols"), j), "must be a column of the stage of its f");
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
    info.name = str(j["name"], "name");
    info.airgroupId = u64(j["airgroupId"], "airgroupId");
    info.airId = u64(j["airId"], "airId");
    info.nBits = u64(j["nBits"], "nBits");
    info.nStages = u64(j["nStages"], "nStages");
    info.nConstants = u64(j["nConstants"], "nConstants");
    info.cmPolsMap = list<PolMapEntry>(j["cmPolsMap"], "cmPolsMap", polMapEntry);
    info.constPolsMap = list<PolMapEntry>(j["constPolsMap"], "constPolsMap", polMapEntry);
    info.challengesMap = list<ChallengeMapEntry>(j["challengesMap"], "challengesMap", challengeMapEntry);
    info.airValuesMap = list<NameStageEntry>(j["airValuesMap"], "airValuesMap", nameStageEntry);
    info.airgroupValuesMap = list<NameStageEntry>(j["airgroupValuesMap"], "airgroupValuesMap", nameStageEntry);
    const json &sections = j["mapSectionsN"];
    if (!sections.is_object()) {
        fail("mapSectionsN", "must be an object");
    }
    for (auto it = sections.begin(); it != sections.end(); ++it) {
        info.mapSectionsN[it.key()] = u64(it.value(), at("mapSectionsN", it.key().c_str()));
    }
    info.openingPoints = list<int64_t>(j["openingPoints"], "openingPoints", i64);
    info.boundaries = list<Boundary>(j["boundaries"], "boundaries", boundary);
    info.evMap = list<EvMapEntry>(j["evMap"], "evMap", evMapEntry);
    info.qDeg = u64(j["qDeg"], "qDeg");
    info.qDim = u64(j["qDim"], "qDim");
    info.maxQDegree = u64(j["maxQDegree"], "maxQDegree");
    info.cExpId = u64(j["cExpId"], "cExpId");
    info.layout = list<LayoutEntry>(j["layout"], "layout", layoutEntry);
    checkIndices(info);
    return info;
}

PilfflonkInfo PilfflonkInfo::load(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw IoError("pilfflonkinfo: cannot open " + path);
    }
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad()) {
        throw IoError("pilfflonkinfo: cannot read " + path);
    }
    try {
        return parse(text);
    } catch (const FormatError &e) {
        throw FormatError(path + ": " + e.what());
    }
}

} // namespace PilFflonk
