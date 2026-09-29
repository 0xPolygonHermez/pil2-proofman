// Tests for PilFflonk::PilfflonkInfo, the C++ side of the Rust → file → C++ round trip of
// <air>.pilfflonkinfo.json (spec §5.2). The fixture is proofman-pilfflonk's: a Rust test
// (pilfflonk/tests/file_types.rs, the_cpp_fixture_is_what_the_rust_types_write) checks that it is
// byte for byte what the Rust type writes for its sample, and this one that the reader gets every
// value of that sample back. Changing the sample means regenerating the fixture and updating both.
#include "pilfflonk_test.hpp"

#include <limits.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "pilfflonk_error.hpp"
#include "pilfflonk_info.hpp"
#include "pilfflonk_test_ptau.hpp"

namespace PilFflonkTest {

namespace {

using PilFflonk::BoundaryType;
using PilFflonk::FormatError;
using PilFflonk::IoError;
using PilFflonk::PilfflonkInfo;
using PilFflonk::PolMapEntry;
using PilFflonk::PolType;

// The test binary is pil2-stark/build/pilfflonkTest (the Makefile's BUILD_DIR), wherever it is run
// from; the fixture is in the workspace's pilfflonk/ crate.
std::string fixturePath() {
    char exe[PATH_MAX];
    const ssize_t length = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    assert(length > 0);
    exe[length] = '\0';
    std::string dir(exe);
    dir = dir.substr(0, dir.rfind('/'));
    return dir + "/../../pilfflonk/tests/fixtures/pilfflonkinfo/Sample.pilfflonkinfo.json";
}

std::string readText(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    assert(file);
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

void expectPol(const PolMapEntry &p, uint64_t stage, const char *name, uint64_t id, uint64_t stageId,
               std::vector<uint64_t> lengths = {}) {
    assert(p.stage == stage && p.name == name && p.dim == 1 && p.polsMapId == id);
    assert(p.stageId == stageId && p.stagePos == stageId && p.lengths == lengths);
}

// Every value of sample_info() in pilfflonk/tests/file_types.rs.
void testReadsWhatRustWrites() {
    const PilfflonkInfo info = PilfflonkInfo::load(fixturePath());

    assert(info.name == "Sample" && info.airgroupId == 0 && info.airId == 0);
    assert(info.nBits == 4 && info.nStages == 2 && info.nConstants == 3 && info.qStage() == 3);

    assert(info.cmPolsMap.size() == 7);
    expectPol(info.cmPolsMap[0], 1, "Sample.a", 0, 0);
    expectPol(info.cmPolsMap[1], 1, "Sample.b", 1, 1);
    expectPol(info.cmPolsMap[2], 1, "Sample.c", 2, 2, {0});
    expectPol(info.cmPolsMap[3], 1, "Sample.c", 3, 3, {1});
    expectPol(info.cmPolsMap[4], 2, "Sample.gsum", 4, 0);
    expectPol(info.cmPolsMap[5], 2, "Sample.im", 5, 1);
    expectPol(info.cmPolsMap[6], 3, "Q0", 6, 0);
    for (size_t i = 0; i < info.cmPolsMap.size(); ++i) {
        const bool im = i == 5;
        assert(info.cmPolsMap[i].imPol == im && info.cmPolsMap[i].hasExpId == im);
    }
    assert(info.cmPolsMap[5].expId == 3);

    assert(info.constPolsMap.size() == 3);
    expectPol(info.constPolsMap[0], 0, "Sample.L1", 0, 0);
    expectPol(info.constPolsMap[1], 0, "Sample.C", 1, 1, {0});
    expectPol(info.constPolsMap[2], 0, "Sample.C", 2, 2, {1});
    for (const PolMapEntry &p : info.constPolsMap) {
        assert(!p.imPol && !p.hasExpId);
    }

    assert(info.challengesMap.size() == 4);
    const char *challenges[] = {"std_alpha", "std_gamma", "std_vc", "std_xi"};
    const uint64_t challengeStages[] = {2, 2, 3, 4};
    const uint64_t challengeStageIds[] = {0, 1, 0, 0};
    for (size_t i = 0; i < 4; ++i) {
        const auto &c = info.challengesMap[i];
        assert(c.name == challenges[i] && c.stage == challengeStages[i] && c.dim == 1);
        assert(c.stageId == challengeStageIds[i]);
    }

    assert(info.airValuesMap.size() == 1 && info.airValuesMap[0].name == "Sample.av");
    assert(info.airValuesMap[0].stage == 1 && info.airValuesMap[0].lengths.empty());
    assert(info.airgroupValuesMap.size() == 1 && info.airgroupValuesMap[0].name == "Sample.gsum_result");
    assert(info.airgroupValuesMap[0].stage == 2);

    const std::map<std::string, uint64_t> sections = {{"const", 3}, {"cm1", 4}, {"cm2", 2}, {"cm3", 1}};
    assert(info.mapSectionsN == sections);
    assert((info.openingPoints == std::vector<int64_t>{-1, 0, 1}));

    assert(info.boundaries.size() == 3);
    assert(info.boundaries[0].type == BoundaryType::EveryRow);
    assert(info.boundaries[1].type == BoundaryType::FirstRow);
    assert(info.boundaries[2].type == BoundaryType::EveryFrame);
    assert(info.boundaries[2].offsetMin == 1 && info.boundaries[2].offsetMax == 2);

    struct Ev {
        PolType type;
        uint64_t id;
        int64_t prime;
        uint64_t openingPos;
    };
    const std::vector<Ev> evMap = {
        {PolType::Cm, 4, -1, 0},   {PolType::Cm, 5, -1, 0},   {PolType::Const, 0, 0, 1}, {PolType::Const, 1, 0, 1},
        {PolType::Const, 2, 0, 1}, {PolType::Cm, 0, 0, 1},    {PolType::Cm, 1, 0, 1},    {PolType::Cm, 2, 0, 1},
        {PolType::Cm, 3, 0, 1},    {PolType::Cm, 4, 0, 1},    {PolType::Cm, 5, 0, 1},    {PolType::Cm, 0, 1, 2},
        {PolType::Cm, 1, 1, 2},
    };
    assert(info.evMap.size() == evMap.size());
    for (size_t i = 0; i < evMap.size(); ++i) {
        const auto &e = info.evMap[i];
        assert(e.type == evMap[i].type && e.id == evMap[i].id && e.prime == evMap[i].prime);
        assert(e.openingPos == evMap[i].openingPos);
    }

    assert(info.qDeg == 2 && info.qDim == 1 && info.maxQDegree == 0 && info.cExpId == 7);

    struct F {
        uint64_t stage;
        std::vector<std::pair<uint64_t, std::string>> pols;
        std::vector<int64_t> offsets;
        uint64_t degree;
    };
    const std::vector<F> layout = {
        {0, {{0, "Sample.L1"}, {1, "Sample.C[0]"}, {2, "Sample.C[1]"}}, {0}, 50},
        {1, {{0, "Sample.a"}, {1, "Sample.b"}}, {0, 1}, 39},
        {1, {{2, "Sample.c[0]"}, {3, "Sample.c[1]"}}, {0}, 37},
        {2, {{4, "Sample.gsum"}, {5, "Sample.im"}}, {-1, 0}, 39},
        {3, {{6, "Q0"}}, {0}, 39},
    };
    assert(info.layout.size() == layout.size());
    for (size_t i = 0; i < layout.size(); ++i) {
        const auto &f = info.layout[i];
        assert(f.stage == layout[i].stage && f.k == layout[i].pols.size());
        assert(f.offsets == layout[i].offsets && f.degree == layout[i].degree);
        assert(f.pols.size() == layout[i].pols.size());
        for (size_t j = 0; j < f.pols.size(); ++j) {
            assert(f.pols[j].id == layout[i].pols[j].first && f.pols[j].name == layout[i].pols[j].second);
        }
    }
}

// The fixture's text with the first `from` replaced by `to`.
std::string mutated(const std::string &text, const std::string &from, const std::string &to) {
    const size_t at = text.find(from);
    assert(at != std::string::npos);
    std::string out = text;
    out.replace(at, from.size(), to);
    return out;
}

void expectFormatError(const std::string &text, const char *fragment) {
    try {
        PilfflonkInfo::parse(text);
    } catch (const FormatError &e) {
        if (std::strstr(e.what(), fragment) == nullptr) {
            std::fprintf(stderr, "expected \"%s\" in: %s\n", fragment, e.what());
            assert(false);
        }
        return;
    }
    std::fprintf(stderr, "no FormatError; expected one with \"%s\"\n", fragment);
    assert(false);
}

void testRefusesWhatIsNotAPilfflonkinfo() {
    const std::string text = readText(fixturePath());
    PilfflonkInfo::parse(text);

    expectFormatError("{", "not valid JSON");
    expectFormatError("[]", "must be an object");
    expectFormatError(mutated(text, " \"cExpId\": 7,\n", ""), "has no field \"cExpId\"");
    expectFormatError(mutated(text, "\"nBits\": 4", "\"nBits\": 4, \"starkStruct\": {}"),
                      "unknown field \"starkStruct\"");
    expectFormatError(mutated(text, "\"nBits\": 4", "\"nBits\": -4"), "nBits: must be an unsigned integer");
    expectFormatError(mutated(text, "\"nBits\": 4", "\"nBits\": 29"), "nBits: must be at most 28");
    expectFormatError(mutated(text, "\"nBits\": 4", "\"nBits\": \"4\""), "nBits: must be an unsigned integer");
    expectFormatError(mutated(text, "\"qDim\": 1", "\"qDim\": 3"), "qDim: must be 1");
    expectFormatError(mutated(text, "\"dim\": 1", "\"dim\": 3"), "cmPolsMap[0].dim: must be 1");
    expectFormatError(mutated(text, "\"nConstants\": 3", "\"nConstants\": 2"), "nConstants");
    expectFormatError(mutated(text, "\"polsMapId\": 1", "\"polsMapId\": 0"), "cmPolsMap[1]");
    expectFormatError(mutated(text, "\"imPol\": true", "\"imPol\": 1"), "cmPolsMap[5].imPol: must be a boolean");
    expectFormatError(mutated(text, "\"type\": \"const\"", "\"type\": \"custom\""), "evMap[2].type");
    expectFormatError(mutated(text, "\"prime\": -1", "\"prime\": -1.5"), "evMap[0].prime: must be an integer");
    expectFormatError(mutated(text, "\"openingPos\": 0", "\"openingPos\": 1"), "evMap[0].openingPos");
    expectFormatError(mutated(text, "\"openingPos\": 0", "\"openingPos\": 7"), "evMap[0].openingPos");
    expectFormatError(mutated(text, "\"name\": \"firstRow\"", "\"name\": \"someRow\""), "is not a boundary");
    expectFormatError(mutated(text, "\"name\": \"everyRow\"", "\"name\": \"everyRow\", \"offsetMin\": 0"),
                      "only everyFrame has offsets");
    expectFormatError(mutated(text, "\"offsetMin\": 1,", ""), "has no field \"offsetMin\"");
    expectFormatError(mutated(text, "\"k\": 3", "\"k\": 2"), "layout[0].k");
    expectFormatError(mutated(text, "\"id\": 6,", "\"id\": 9,"), "layout[4].pols[0]");
    expectFormatError(mutated(text, "\"id\": 6,", "\"id\": 5,"), "layout[4].pols[0]");
    expectFormatError(mutated(text, "\"stage\": 3,\n   \"pols\"", "\"stage\": 4,\n   \"pols\""), "layout[4].stage");
    expectFormatError(mutated(text, "\"stage\": 3,", "\"stage\": 4,"), "cmPolsMap[6]");
    expectFormatError(mutated(text, "\"stagePos\": 3", "\"stagePos\": 4"), "cmPolsMap[3].stagePos");
    expectFormatError(mutated(text, "\"offsets\": [\n    0\n   ],", "\"offsets\": [],"), "layout[0].offsets");
}

void testLoadNamesTheFile() {
    bool threw = false;
    try {
        PilfflonkInfo::load("/nonexistent/pilfflonk_info_test/Sample.pilfflonkinfo.json");
    } catch (const IoError &e) {
        threw = std::strstr(e.what(), "/nonexistent/pilfflonk_info_test/Sample.pilfflonkinfo.json") != nullptr;
    }
    assert(threw);

    TestDir dir;
    const std::string path = dir.file("Broken.pilfflonkinfo.json");
    {
        std::ofstream out(path, std::ios::binary);
        out << mutated(readText(fixturePath()), "\"qDim\": 1", "\"qDim\": 3");
    }
    threw = false;
    try {
        PilfflonkInfo::load(path);
    } catch (const FormatError &e) {
        threw = std::strstr(e.what(), path.c_str()) != nullptr && std::strstr(e.what(), "qDim") != nullptr;
    }
    assert(threw);

    const std::string empty = dir.file("Empty.pilfflonkinfo.json");
    { std::ofstream out(empty, std::ios::binary); }
    threw = false;
    try {
        PilfflonkInfo::load(empty);
    } catch (const FormatError &) {
        threw = true;
    }
    assert(threw);
}

} // namespace

void runInfoTests() {
    testReadsWhatRustWrites();
    testRefusesWhatIsNotAPilfflonkinfo();
    testLoadNamesTheFile();
}

} // namespace PilFflonkTest
