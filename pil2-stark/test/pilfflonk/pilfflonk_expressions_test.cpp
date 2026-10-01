// Tests for PilFflonk::ExpressionsBin and PilFflonk::Expressions, the reader and the interpreter of
// <air>.bin (pilfflonk/docs/formats.md#bytecode). The fixtures are pilfflonk-setup's, in
// setup/pilfflonk/tests/fixtures/bytecode/, written by its Rust tests
// (setup/pilfflonk/tests/bytecode.rs and bytecode/interpreter.rs), which check that they are what
// they compute:
//
// - Sample.bin: the bytecode of sample() in bytecode.rs, whose every field testReadsTheRustSample
//   checks, its hints (revision 3) too; Sample.expected.json: inputs for its codes, and what the
//   Rust evaluator (num-bigint) gives them.
// - fibonacci/: the Fibonacci (pilfflonk/tests/data/fibonacci.rs) as setup-pilfflonk writes it, its
//   witness, the qVerifier as a bytecode, and the oracle's evaluations and Q at a point.
//
// Changing them means regenerating them (PILFFLONK_UPDATE_FIXTURES=1, see those files) and updating
// these tests. The fixtures are found from the test binary, pil2-stark/build/pilfflonkTest, or from
// $PILFFLONK_REPO_ROOT if it is set.
#include "pilfflonk_test.hpp"

#include <limits.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pilfflonk_error.hpp"
#include "pilfflonk_expressions.hpp"
#include "pilfflonk_expressions_bin.hpp"
#include "pilfflonk_fr.hpp"
#include "pilfflonk_info.hpp"
#include "pilfflonk_lde.hpp"

namespace PilFflonkTest {

namespace {

using json = nlohmann::json;
using PilFflonk::Boundary;
using PilFflonk::BoundaryType;
using PilFflonk::Expressions;
using PilFflonk::ExpressionsBin;
using PilFflonk::ExpressionsDomain;
using PilFflonk::FormatError;
using PilFflonk::FrElement;
using PilFflonk::Hint;
using PilFflonk::HintFieldValue;
using PilFflonk::HintOp;
using PilFflonk::IoError;
using PilFflonk::Lde;
using PilFflonk::ParserArgs;
using PilFflonk::ParserParams;
using PilFflonk::PilfflonkInfo;
using PilFflonk::PointValues;
using PilFflonk::ProverValues;
using Engine = AltBn128::Engine;

Engine::Fr &F() { return Engine::engine.fr; }

std::string repoPath(const std::string &relative) {
    if (const char *root = std::getenv("PILFFLONK_REPO_ROOT")) {
        return std::string(root) + "/" + relative;
    }
    char exe[PATH_MAX];
    const ssize_t length = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    assert(length > 0);
    exe[length] = '\0';
    std::string dir(exe);
    dir = dir.substr(0, dir.rfind('/'));
    return dir + "/../../" + relative;
}

std::string fixture(const std::string &name) { return repoPath("setup/pilfflonk/tests/fixtures/bytecode/" + name); }

std::vector<uint8_t> readBytes(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    assert(file);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

json readJson(const std::string &path) {
    const std::vector<uint8_t> bytes = readBytes(path);
    return json::parse(bytes.begin(), bytes.end());
}

FrElement fr(const std::string &decimal) {
    FrElement e;
    F().fromString(e, decimal, 10);
    return e;
}

FrElement fr(const json &value) { return fr(value.get<std::string>()); }

FrElement fromUI(uint64_t v) {
    FrElement e;
    F().fromUI(e, v);
    return e;
}

std::vector<FrElement> frs(const json &values) {
    std::vector<FrElement> out;
    for (const json &v : values) out.push_back(fr(v));
    return out;
}

bool eq(const FrElement &a, const FrElement &b) { return F().eq(a, b); }

FrElement inv(const FrElement &a) {
    FrElement r;
    F().inv(r, a);
    return r;
}

FrElement power(const FrElement &base, uint64_t exponent) {
    FrElement r = F().one();
    for (uint64_t i = 0; i < exponent; ++i) F().mul(r, r, base);
    return r;
}

// A canonical 32-byte little-endian value, into Montgomery form.
FrElement fromLE(const uint8_t *bytes) {
    FrElement e;
    F().fromRprLE(e, bytes, 32);
    return e;
}

template <typename Exception, typename Call>
std::string thrown(Call call) {
    try {
        call();
    } catch (const Exception &e) {
        return e.what();
    }
    assert(false && "no exception");
    return "";
}

bool contains(const std::string &text, const std::string &part) { return text.find(part) != std::string::npos; }

std::vector<Boundary> boundariesOf(const json &boundaries) {
    std::vector<Boundary> out;
    for (const json &b : boundaries) {
        const std::string name = b["name"];
        if (name == "everyRow") out.push_back({BoundaryType::EveryRow, 0, 0});
        else if (name == "firstRow") out.push_back({BoundaryType::FirstRow, 0, 0});
        else if (name == "lastRow") out.push_back({BoundaryType::LastRow, 0, 0});
        else out.push_back({BoundaryType::EveryFrame, b["offsetMin"].get<uint64_t>(), b["offsetMax"].get<uint64_t>()});
    }
    return out;
}

// ---------------------------------------------------------------------------------------------
// The reader
// ---------------------------------------------------------------------------------------------

using Op = std::array<uint32_t, 8>;

// The entry's fields, and its ops, 8 args each, as sample() in bytecode.rs writes them.
void expectEntry(const ParserParams &p, const ParserArgs &args, uint32_t stage, uint32_t nTemp, uint32_t opsOffset,
                 const std::string &line, const std::vector<Op> &ops) {
    assert(p.stage == stage && p.nTemp == nTemp && p.destId == 0 && p.line == line);
    assert(p.nOps == ops.size() && p.opsOffset == opsOffset);
    assert(p.nArgs == 8 * ops.size() && p.argsOffset == 8 * opsOffset);
    for (size_t k = 0; k < ops.size(); ++k) {
        assert(args.ops[opsOffset + k] == 0);
        for (size_t w = 0; w < 8; ++w) {
            assert(args.args[p.argsOffset + 8 * k + w] == ops[k][w]);
        }
    }
}

// Every field of Sample.bin: sample() of setup/pilfflonk/tests/bytecode.rs, for nStages = 2, so
// Zi is type 4, and the tmp buffer is 6: public 8, number 9, air value 10, proof value 11, airgroup
// value 12, challenge 13, evaluation 14. Openings 0 … 4 are −2 … 2.
void testReadsTheRustSample() {
    const ExpressionsBin bin = ExpressionsBin::load(fixture("Sample.bin"));
    assert(bin.nStages == 2 && bin.maxTmp == 3 && bin.maxArgs == 64 && bin.maxOps == 8);
    assert(bin.types().zi() == 4 && bin.types().tmp() == 6 && bin.types().evals() == 14);

    const std::string R_MINUS_ONE = "21888242871839275222246405745257275088548364400416034343698204186575808495616";
    const std::string WIDE = "14474011154664524427946373126085988481658748083205070504932198000989141204997"; // 2^253 + 5
    auto numbers = [](const ParserArgs &args) {
        std::vector<std::string> out;
        for (const FrElement &n : args.numbers) out.push_back(F().toString(n));
        return out;
    };

    const ParserArgs &e = bin.expressionsBinArgsExpressions;
    assert(bin.expressionsInfo.size() == 4 && e.ops.size() == 16 && e.args.size() == 128);
    assert((numbers(e) == std::vector<std::string>{R_MINUS_ONE, WIDE, "0"}));
    expectEntry(bin.expression(3), e, 1, 2, 0, "Sample.ImPol",
                {Op{2, 0, 1, 0, 3, 1, 1, 2}, Op{0, 1, 0, 0, 1, 6, 0, 0}, Op{1, 0, 6, 1, 0, 9, 0, 0}});
    expectEntry(bin.expression(7), e, 3, 3, 3, "",
                {Op{2, 0, 2, 0, 2, 13, 0, 0}, Op{1, 1, 8, 1, 0, 10, 0, 0}, Op{0, 0, 6, 0, 0, 6, 1, 0},
                 Op{0, 1, 9, 1, 0, 9, 2, 0}, Op{2, 0, 6, 0, 0, 6, 1, 0}, Op{2, 2, 4, 2, 0, 1, 1, 4},
                 Op{0, 0, 6, 0, 0, 6, 2, 0}, Op{2, 0, 4, 1, 0, 6, 0, 0}});
    expectEntry(bin.expression(9), e, 1, 2, 11, "a hint's expression",
                {Op{3, 0, 11, 1, 0, 12, 0, 0}, Op{2, 1, 6, 0, 0, 8, 0, 0}, Op{2, 0, 6, 1, 0, 9, 0, 0}});
    expectEntry(bin.expression(11), e, 0, 1, 14, "verifier form", {Op{2, 0, 14, 0, 0, 14, 3, 0}, Op{0, 0, 4, 1, 0, 6, 0, 0}});

    const ParserArgs &c = bin.expressionsBinArgsConstraints;
    const std::vector<ParserParams> &cs = bin.constraintsInfoDebug;
    assert(cs.size() == 4 && c.ops.size() == 11 && c.args.size() == 88);
    assert((numbers(c) == std::vector<std::string>{WIDE, R_MINUS_ONE, "0"}));
    const uint32_t rows[4][3] = {{0, 8, 0}, {0, 1, 0}, {7, 8, 0}, {1, 6, 1}};
    for (size_t i = 0; i < 4; ++i) {
        assert(cs[i].firstRow == rows[i][0] && cs[i].lastRow == rows[i][1] && cs[i].imPol == (rows[i][2] == 1));
    }
    expectEntry(cs[0], c, 1, 1, 0, "sample.pil:10 a' - a * b === 0",
                {Op{2, 0, 1, 0, 2, 1, 1, 2}, Op{1, 0, 1, 0, 3, 6, 0, 0}});
    expectEntry(cs[1], c, 1, 1, 2, "sample.pil:11 L1 * (a - in1) === 0",
                {Op{1, 0, 1, 0, 2, 8, 0, 0}, Op{2, 0, 0, 0, 2, 6, 0, 0}});
    expectEntry(cs[2], c, 1, 1, 4, "sample.pil:12 b - (2**253 + 5) === 0", {Op{1, 0, 1, 1, 2, 9, 0, 0}});
    expectEntry(cs[3], c, 2, 2, 5, "Sample.ImPol",
                {Op{2, 0, 1, 0, 3, 1, 1, 2}, Op{0, 0, 0, 0, 1, 6, 0, 0}, Op{1, 0, 6, 0, 0, 9, 1, 0},
                 Op{1, 0, 1, 2, 2, 6, 0, 0}, Op{0, 1, 2, 0, 0, 9, 2, 0}, Op{0, 0, 6, 0, 0, 6, 1, 0}});
    assert(contains(thrown<std::invalid_argument>([&] { bin.expression(4); }), "no expression 4"));

    // The hints: sample_hints() of bytecode.rs. Openings −1, 0 and 2 are indices 1, 2 and 4.
    assert(bin.hints.size() == 2);
    assert(bin.hintIds("gsum_col") == std::vector<uint64_t>{0});
    assert(bin.hintIds("gprod_col") == std::vector<uint64_t>{1});
    assert(bin.hintIds("im_col").empty());
    auto expectValue = [&](const HintFieldValue &v, HintOp op, uint64_t id, const std::vector<uint64_t> &pos) {
        assert(v.op == op && v.pos == pos);
        if (op != HintOp::Number && op != HintOp::String) assert(v.id == id);
    };
    auto only = [&](const Hint &h, size_t f, const std::string &name) -> const HintFieldValue & {
        assert(h.fields[f].name == name && h.fields[f].values.size() == 1);
        return h.fields[f].values[0];
    };
    const Hint &gsum = bin.hints[0];
    assert(gsum.name == "gsum_col" && gsum.fields.size() == 6);
    expectValue(only(gsum, 0, "reference"), HintOp::Cm, 3, {});
    assert(only(gsum, 0, "reference").rowOffsetIndex == 2);
    expectValue(only(gsum, 1, "numerator_air"), HintOp::Tmp, 9, {});
    expectValue(only(gsum, 2, "denominator_air"), HintOp::Const, 0, {});
    assert(only(gsum, 2, "denominator_air").rowOffsetIndex == 1);
    const char *numbers3[3][2] = {{"numerator_direct", "0"}, {"denominator_direct", "1"}, {"result", "0"}};
    for (size_t f = 3; f < 6; ++f) {
        const HintFieldValue &v = only(gsum, f, numbers3[f - 3][0]);
        assert(v.op == HintOp::Number && F().toString(v.value) == numbers3[f - 3][1]);
    }
    const Hint &gprod = bin.hints[1];
    assert(gprod.name == "gprod_col" && gprod.fields.size() == 6);
    expectValue(only(gprod, 0, "reference"), HintOp::Cm, 3, {});
    assert(only(gprod, 0, "reference").rowOffsetIndex == 4);
    const std::vector<HintFieldValue> &terms = gprod.fields[1].values;
    assert(gprod.fields[1].name == "terms" && terms.size() == 4);
    expectValue(terms[0], HintOp::Challenge, 1, {0, 0});
    expectValue(terms[1], HintOp::Public, 0, {0, 1});
    expectValue(terms[2], HintOp::AirValue, 0, {1, 0});
    expectValue(terms[3], HintOp::ProofValue, 1, {1, 1});
    const std::vector<HintFieldValue> &names = gprod.fields[2].values;
    assert(gprod.fields[2].name == "names" && names.size() == 2);
    expectValue(names[0], HintOp::String, 0, {0});
    expectValue(names[1], HintOp::String, 0, {1});
    assert(names[0].stringValue == "Sample" && names[1].stringValue.empty());
    assert(F().toString(only(gprod, 3, "wide").value) == WIDE);
    expectValue(only(gprod, 4, "group"), HintOp::AirgroupValue, 0, {});
    assert(F().toString(only(gprod, 5, "result").value) == R_MINUS_ONE);
}

// What the Rust reader refuses, the C++ one refuses too.
void testRefusesWhatIsNotRevision3() {
    const std::vector<uint8_t> bytes = readBytes(fixture("Sample.bin"));
    auto refused = [&](std::vector<uint8_t> b, const std::string &why) {
        const std::string message = thrown<FormatError>([&] { ExpressionsBin::parse(b.data(), b.size(), "Sample.bin"); });
        if (!contains(message, why)) {
            std::fprintf(stderr, "expected \"%s\" in \"%s\"\n", why.c_str(), message.c_str());
            assert(false);
        }
    };
    auto patched = [&](size_t at, std::vector<uint8_t> with) {
        std::vector<uint8_t> b = bytes;
        std::copy(with.begin(), with.end(), b.begin() + at);
        return b;
    };
    const size_t section1 = 12 + 12; // after the file's header and the section's
    refused(patched(4, {1, 0, 0, 0}), "version 0x1,");  // a STARK chps
    refused(patched(0, {'z', 'k', 'e', 'y'}), "not a \"chps\"");
    refused(patched(section1, {2, 0, 0, 0}), "section 1 starts with");
    refused(patched(section1 + 8, {0}), "section 1 starts with"); // r
    refused(std::vector<uint8_t>(bytes.begin(), bytes.end() - 1), "ends before");
    std::vector<uint8_t> longer = bytes;
    longer.push_back(0);
    refused(longer, "after its content");
    refused(patched(section1 + 44, {9, 0, 0, 0}), "maxTmp, maxArgs and maxOps"); // maxTmp
    // A revision-2 file: its version, and its prefix's.
    std::vector<uint8_t> revision2 = patched(4, {2, 0, 0x66, 0x70});
    std::copy_n(revision2.begin() + 4, 4, revision2.begin() + section1);
    refused(revision2, "version 0x70660002, and pilfflonk's is 0x70660003: not a pilfflonk bytecode of revision 3");

    // The hints, section 3, the last of the file: its last value is the number r − 1 of no position.
    const size_t tail = bytes.size() - 4 - 32;
    refused(patched(tail, std::vector<uint8_t>(PilFflonk::FR_MODULUS_LE, PilFflonk::FR_MODULUS_LE + 32)),
            "hint 1 (`gprod_col`), field `result`: a number not below r");
    const std::string number = "number";
    assert(std::string(bytes.begin() + tail - 7, bytes.begin() + tail - 1) == number);
    refused(patched(tail - 7, {'c', 'u', 's', 't', 'o', 'm'}), "unknown value kind \"custom\"");
    refused(patched(bytes.size() - 4, {1, 0, 0, 0}), "the hints: it ends before its content does"); // nPos 1
    // The numerator of the first hint, expression 9, becomes one section 1 does not have.
    const std::vector<uint8_t> tmp9 = {'t', 'm', 'p', 0, 9, 0, 0, 0};
    const size_t at = std::search(bytes.begin(), bytes.end(), tmp9.begin(), tmp9.end()) - bytes.begin();
    assert(at < bytes.size());
    refused(patched(at + 4, {10, 0, 0, 0}), "expression 10, which section 1 does not have");

    // The first op of section 1: after the prefix, the counts and the four entries.
    size_t pos = section1 + 44 + 28;
    for (int i = 0; i < 4; ++i) {
        pos += 4 * 8;
        while (bytes[pos] != 0) ++pos;
        ++pos;
    }
    refused(patched(pos, {1}), "is of dimensions 1");
    const size_t args = pos + 16; // after the 16 ops
    refused(patched(args, {4, 0, 0, 0}), "unknown opType 4");
    refused(patched(args + 8, {0xe8, 0x03, 0, 0}), "is no operand of an AIR of 2 stages");  // aType 1000
    assert(contains(thrown<IoError>([] { ExpressionsBin::load(fixture("no/such/file.bin")); }), "cannot open"));
}

// ---------------------------------------------------------------------------------------------
// The interpreter on Sample.bin, against the Rust evaluator
// ---------------------------------------------------------------------------------------------

// Columns of the JSON, [type][pos][point], and the pointers ProverValues takes.
struct Columns {
    std::vector<std::vector<std::vector<FrElement>>> values;
    std::vector<std::vector<const FrElement *>> pointers;

    explicit Columns(const json &j) {
        for (const json &type : j) {
            values.emplace_back();
            for (const json &column : type) values.back().push_back(frs(column));
        }
        for (const auto &type : values) {
            pointers.emplace_back();
            for (const auto &column : type) pointers.back().push_back(column.data());
        }
    }
};

void expectValues(const std::vector<FrElement> &got, const json &expected, const char *what) {
    assert(got.size() == expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (!eq(got[i], fr(expected[i]))) {
            std::fprintf(stderr, "%s, point %zu: %s, and the Rust evaluator's is %s\n", what, i,
                         F().toString(got[i]).c_str(), expected[i].get<std::string>().c_str());
            assert(false);
        }
    }
}

void testSampleGivesTheRustEvaluatorsValues() {
    const json j = readJson(fixture("Sample.expected.json"));
    const ExpressionsBin bin = ExpressionsBin::load(fixture("Sample.bin"));
    PilfflonkInfo info{};
    info.nBits = j["nBits"];
    info.nStages = 2;
    info.openingPoints = j["openingPoints"].get<std::vector<int64_t>>();
    info.boundaries = boundariesOf(j["boundaries"]);
    const Expressions expressions(bin, info);
    const json &expected = j["expected"];
    const uint64_t nBits = j["nBits"], nBitsExt = j["nBitsExt"];

    auto proverValues = [&](const Columns &columns) {
        ProverValues v;
        v.columns = columns.pointers;
        v.publics = frs(j["publics"]);
        v.challenges = frs(j["challenges"]);
        v.airValues = frs(j["airValues"]);
        v.proofValues = frs(j["proofValues"]);
        v.airgroupValues = frs(j["airgroupValues"]);
        return v;
    };

    // On H: the intermediate polynomial (3), the value (9) and the constraints.
    const Columns traceColumns(j["trace"]);
    const ProverValues onTrace = proverValues(traceColumns);
    const ExpressionsDomain trace = ExpressionsDomain::trace(nBits);
    std::vector<FrElement> out(trace.size());
    expressions.calculateExpression(3, trace, onTrace, out.data());
    expectValues(out, expected["expressions"]["3"], "expression 3 on H");
    expressions.calculateExpression(9, trace, onTrace, out.data());
    expectValues(out, expected["expressions"]["9"], "expression 9 on H");
    for (uint64_t c = 0; c < 4; ++c) {
        expressions.calculateConstraint(c, trace, onTrace, out.data());
        expectValues(out, expected["constraints"][c], "a constraint on H");
    }

    // On the coset: Zi, and Q (7).
    const Columns cosetColumns(j["coset"]);
    const ExpressionsDomain coset = ExpressionsDomain::coset(nBits, nBitsExt, info.boundaries);
    assert(coset.size() == 16 && coset.extendBits() == 1 && coset.nZerofiers() == 4);
    for (uint64_t b = 0; b < 4; ++b) {
        expectValues(coset.zerofier(b), expected["cosetZerofiers"][b], "Zi on the coset");
    }
    out.resize(coset.size());
    expressions.calculateExpression(7, coset, proverValues(cosetColumns), out.data());
    expectValues(out, expected["expressions"]["7"], "expression 7 on the coset");

    // At ξ: Zi, and the code of evaluations (11).
    const FrElement xi = fr(j["xi"]);
    PointValues point;
    point.zerofiers = PilFflonk::zerofiersAt(nBits, info.boundaries, xi);
    expectValues(point.zerofiers, expected["xiZerofiers"], "Zi at ξ");
    point.evals = frs(j["evals"]);
    point.publics = frs(j["publics"]);
    point.challenges = frs(j["challenges"]);
    assert(eq(expressions.evaluateExpressionAt(11, point), fr(expected["expressions"]["11"])));

    // Each mode refuses what it does not have, before it writes anything.
    std::vector<FrElement> untouched(trace.size(), F().one());
    assert(contains(thrown<std::invalid_argument>(
                        [&] { expressions.calculateExpression(7, trace, onTrace, untouched.data()); }),
                    "which the domain does not have (on H, none)"));
    assert(contains(thrown<std::invalid_argument>([&] { expressions.evaluateExpressionAt(3, point); }),
                    "reads a column"));
    assert(contains(thrown<std::invalid_argument>([&] {
                        expressions.calculateExpression(11, coset, proverValues(cosetColumns), out.data());
                    }),
                    "only the verifier mode has"));
    ProverValues noPublics = onTrace;
    noPublics.publics.clear();
    assert(contains(thrown<std::invalid_argument>(
                        [&] { expressions.calculateExpression(9, trace, noPublics, untouched.data()); }),
                    "public 0, of 0"));
    ProverValues noColumn = onTrace;
    noColumn.columns[1][1] = nullptr;
    assert(contains(thrown<std::invalid_argument>(
                        [&] { expressions.calculateExpression(3, trace, noColumn, untouched.data()); }),
                    "column 1 of stage 1"));
    for (const FrElement &v : untouched) assert(eq(v, F().one()));

    // A pilfflonkinfo that does not fit the bytecode.
    PilfflonkInfo fewer = info;
    fewer.openingPoints.pop_back();
    assert(contains(thrown<FormatError>([&] { (void)Expressions(bin, fewer); }), "at opening point 4"));
    PilfflonkInfo oneBoundary = info;
    oneBoundary.boundaries.resize(1);
    assert(contains(thrown<FormatError>([&] { (void)Expressions(bin, oneBoundary); }), "Zi of boundary 1"));
    PilfflonkInfo oneStage = info;
    oneStage.nStages = 1;
    assert(contains(thrown<FormatError>([&] { (void)Expressions(bin, oneStage); }), "of 2 stages"));
}

// ---------------------------------------------------------------------------------------------
// Zi
// ---------------------------------------------------------------------------------------------

// Zi on the coset against a direct computation of its closed forms
// (pilfflonk/docs/protocol.md#constraint-polynomial), one inversion per point, for every kind of
// boundary, with ffiasm's FFT's roots.
void testZerofiersOnTheCoset() {
    const uint64_t nBits = 3, nBitsExt = 5, n = 8, m = 32;
    FFT<Engine::Fr> fft(m);
    const FrElement w = fft.root(nBitsExt, 1);
    const FrElement wN = fft.root(nBits, 1);
    assert(eq(PilFflonk::rootOfUnity(nBitsExt), w) && eq(PilFflonk::rootOfUnity(nBits), wN));
    // ω_{2^28} has order 2^28: squared 27 times it is −1.
    FrElement w28 = PilFflonk::rootOfUnity(28);
    for (int i = 0; i < 27; ++i) F().mul(w28, w28, w28);
    assert(eq(w28, fr(std::string("21888242871839275222246405745257275088548364400416034343698204186575808495616"))));

    const std::vector<Boundary> boundaries = {{BoundaryType::EveryRow, 0, 0},
                                              {BoundaryType::FirstRow, 0, 0},
                                              {BoundaryType::LastRow, 0, 0},
                                              {BoundaryType::EveryFrame, 1, 2},
                                              {BoundaryType::EveryFrame, 0, 0}};
    const ExpressionsDomain coset = ExpressionsDomain::coset(nBits, nBitsExt, boundaries);
    assert(coset.size() == m && coset.extendBits() == 2 && coset.nZerofiers() == boundaries.size());

    FrElement x = fromUI(5); // g
    const FrElement lastRoot = power(wN, n - 1);
    bool lastIsNotFirst = false;
    for (uint64_t i = 0; i < m; ++i) {
        FrElement zh, d;
        F().sub(zh, power(x, n), F().one());
        const FrElement everyRow = inv(zh);
        F().sub(d, x, F().one());
        const FrElement firstRow = F().mul(zh, inv(d));
        // X − ω^(N−1), not X − ω^N = X − 1 (pilfflonk/docs/README.md#stark-lastrow-zerofier)
        F().sub(d, x, lastRoot);
        const FrElement lastRow = F().mul(zh, inv(d));
        FrElement frame = F().one();
        for (uint64_t j : {uint64_t(0), n - 1, n - 2}) {
            F().sub(d, x, power(wN, j));
            F().mul(frame, frame, d);
        }
        assert(eq(coset.zerofier(0)[i], everyRow));
        assert(eq(coset.zerofier(1)[i], firstRow));
        assert(eq(coset.zerofier(2)[i], lastRow));
        assert(eq(coset.zerofier(3)[i], frame));
        assert(eq(coset.zerofier(4)[i], F().one())); // an everyFrame that excludes nothing: Z_D = Z_H
        lastIsNotFirst = lastIsNotFirst || !eq(lastRow, firstRow);
        // The same at the point, by zerofiersAt.
        const std::vector<FrElement> at = PilFflonk::zerofiersAt(nBits, boundaries, x);
        for (size_t b = 0; b < boundaries.size(); ++b) assert(eq(at[b], coset.zerofier(b)[i]));
        F().mul(x, x, w);
    }
    assert(lastIsNotFirst);

    // H, and domains and boundaries there are none of.
    assert(ExpressionsDomain::trace(nBits).nZerofiers() == 0 && ExpressionsDomain::trace(nBits).size() == n);
    assert(contains(thrown<std::invalid_argument>([&] { PilFflonk::zerofiersAt(nBits, boundaries, power(wN, 3)); }),
                    "in H"));
    assert(contains(thrown<std::invalid_argument>([&] { ExpressionsDomain::coset(4, 3, boundaries); }),
                    "nBits <= nBitsExt <= 28"));
    assert(contains(thrown<std::invalid_argument>([&] { ExpressionsDomain::coset(3, 29, boundaries); }),
                    "nBits <= nBitsExt <= 28"));
    assert(contains(thrown<std::invalid_argument>(
                        [&] { ExpressionsDomain::coset(3, 4, {{BoundaryType::EveryFrame, 5, 4}}); }),
                    "everyFrame excludes 5 + 4 rows of 8"));
    assert(contains(thrown<std::invalid_argument>([] { PilFflonk::rootOfUnity(29); }), "order 2^29"));

    // The coset in parts (pilfflonk/docs/protocol.md#q-in-parts): point i of part p of 2^partBits
    // points is point p + (N'/S)·i of the coset, Zi and all, bit for bit, and a column at an opening
    // point is read as many rows later in the part as on the coset (extendBits).
    for (uint64_t partBits = nBits; partBits <= nBitsExt; ++partBits) {
        const uint64_t S = uint64_t(1) << partBits, nParts = m / S;
        for (uint64_t part = 0; part < nParts; ++part) {
            const ExpressionsDomain piece = ExpressionsDomain::cosetPart(nBits, nBitsExt, partBits, part, boundaries);
            assert(piece.size() == S && piece.nBits() == nBits && piece.extendBits() == partBits - nBits);
            assert(piece.nZerofiers() == boundaries.size());
            for (size_t b = 0; b < boundaries.size(); ++b) {
                for (uint64_t i = 0; i < S; ++i) {
                    const FrElement &whole = coset.zerofier(b)[part + nParts * i];
                    assert(std::memcmp(&piece.zerofier(b)[i], &whole, sizeof(FrElement)) == 0);
                }
            }
        }
    }
    assert(contains(thrown<std::invalid_argument>([&] { ExpressionsDomain::cosetPart(3, 5, 2, 0, boundaries); }),
                    "nBits <= partBits <= nBitsExt <= 28"));
    assert(contains(thrown<std::invalid_argument>([&] { ExpressionsDomain::cosetPart(3, 5, 6, 0, boundaries); }),
                    "nBits <= partBits <= nBitsExt <= 28"));
    assert(contains(thrown<std::invalid_argument>([&] { ExpressionsDomain::cosetPart(3, 5, 4, 2, boundaries); }),
                    "part 2 of the 2 of 2^4 points"));
    assert(contains(thrown<std::invalid_argument>(
                        [&] { ExpressionsDomain::cosetPart(3, 4, 3, 0, {{BoundaryType::EveryFrame, 5, 4}}); }),
                    "everyFrame excludes 5 + 4 rows of 8"));
}

// ---------------------------------------------------------------------------------------------
// The Fibonacci
// ---------------------------------------------------------------------------------------------

struct Fibonacci {
    PilfflonkInfo info = PilfflonkInfo::load(fixture("fibonacci/Fibonacci.pilfflonkinfo.json"));
    ExpressionsBin bin = ExpressionsBin::load(fixture("fibonacci/Fibonacci.bin"));
    json oracle = readJson(fixture("fibonacci/Fibonacci.oracle.json"));
    uint64_t n = uint64_t(1) << info.nBits;
    // The stage-1 columns on H by stagePos, the witness's (l1, l2) and then the im pol's; and the
    // fixed ones.
    std::vector<std::vector<FrElement>> stage1;
    std::vector<std::vector<FrElement>> fixed;
    uint64_t imPos = 0;
    uint64_t imExpId = 0;

    Fibonacci() {
        const std::vector<uint8_t> witness = readBytes(fixture("fibonacci/Fibonacci.witness.bin"));
        const std::vector<uint8_t> constants = readBytes(fixture("fibonacci/Fibonacci.const"));
        const uint64_t nWitness = witness.size() / (32 * n), nFixed = info.nConstants;
        assert(witness.size() == 32 * n * nWitness && constants.size() == 32 * n * nFixed);
        stage1.assign(info.mapSectionsN.at("cm1"), std::vector<FrElement>(n));
        fixed.assign(nFixed, std::vector<FrElement>(n));
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t c = 0; c < nWitness; ++c) stage1[c][i] = fromLE(&witness[32 * (i * nWitness + c)]);
            for (uint64_t c = 0; c < nFixed; ++c) fixed[c][i] = fromLE(&constants[32 * (i * nFixed + c)]);
        }
        for (const auto &p : info.cmPolsMap) {
            if (p.imPol) {
                imPos = p.stagePos;
                imExpId = p.expId;
            }
        }
        assert(info.nStages == 1 && nWitness == 2 && imPos == 2 && imExpId == 6);
    }

    ProverValues values(std::vector<std::vector<FrElement>> &fixedColumns,
                        std::vector<std::vector<FrElement>> &stage1Columns) const {
        ProverValues v;
        v.columns.resize(2);
        for (auto &c : fixedColumns) v.columns[0].push_back(c.data());
        for (auto &c : stage1Columns) v.columns[1].push_back(c.data());
        v.publics = frs(oracle["publics"]);
        v.challenges = frs(oracle["challenges"]);
        return v;
    }
};

// The smallest nBitsExt whose domain holds every f of the layout
// (pilfflonk/docs/protocol.md#degrees): Q's bound.
uint64_t nBitsExtOf(const PilfflonkInfo &info) {
    uint64_t degree = 0;
    for (const auto &f : info.layout) degree = std::max(degree, f.degree);
    uint64_t bits = 0;
    while ((uint64_t(1) << bits) < degree) ++bits;
    return bits;
}

// Q of a witness: its im pol on H with the bytecode, every column extended to the coset, Q there
// with the bytecode, and back to coefficients.
std::vector<FrElement> qCoefficients(const Fibonacci &fib, const Expressions &expressions,
                                     std::vector<std::vector<FrElement>> stage1) {
    const ExpressionsDomain trace = ExpressionsDomain::trace(fib.info.nBits);
    std::vector<std::vector<FrElement>> fixed = fib.fixed;
    expressions.calculateExpression(fib.imExpId, trace, fib.values(fixed, stage1), stage1[fib.imPos].data());

    const uint64_t nBitsExt = nBitsExtOf(fib.info);
    const Lde lde(fib.info.nBits, nBitsExt);
    const uint64_t m = lde.extendedSize();
    std::vector<std::vector<FrElement>> fixedCoset, stage1Coset;
    auto extend = [&](std::vector<std::vector<FrElement>> &columns, std::vector<std::vector<FrElement>> &out) {
        for (auto &column : columns) {
            std::vector<FrElement> coefs(fib.n), evals(m);
            FrElement *e = column.data(), *c = coefs.data(), *x = evals.data();
            lde.intt(&e, &c, 1);
            const FrElement *cc = coefs.data();
            lde.extendCoset(&cc, &x, 1, fib.n);
            out.push_back(std::move(evals));
        }
    };
    extend(fixed, fixedCoset);
    extend(stage1, stage1Coset);

    const ExpressionsDomain coset = ExpressionsDomain::coset(fib.info.nBits, nBitsExt, fib.info.boundaries);
    std::vector<FrElement> q(m);
    expressions.calculateExpression(fib.info.cExpId, coset, fib.values(fixedCoset, stage1Coset), q.data());
    const FrElement *qe = q.data();
    FrElement *qc = q.data();
    lde.interpolateCoset(&qe, &qc, 1);
    return q;
}

// The first coefficient from `from` on that is not 0, or q.size().
uint64_t firstNonZeroFrom(const std::vector<FrElement> &q, uint64_t from) {
    for (uint64_t j = from; j < q.size(); ++j) {
        if (!F().isZero(q[j])) return j;
    }
    return q.size();
}

void testTheFibonaccisQIsAPolynomialOfItsDegree() {
    const Fibonacci fib;
    const Expressions expressions(fib.bin, fib.info);
    const uint64_t nBitsExt = nBitsExtOf(fib.info);
    assert(fib.n == 256 && nBitsExt == 9 && fib.info.qDeg == 1);

    // The witness satisfies every constraint on its rows, its im pol's included.
    {
        std::vector<std::vector<FrElement>> stage1 = fib.stage1, fixed = fib.fixed;
        const ExpressionsDomain trace = ExpressionsDomain::trace(fib.info.nBits);
        const ProverValues v = fib.values(fixed, stage1);
        expressions.calculateExpression(fib.imExpId, trace, v, stage1[fib.imPos].data());
        std::vector<FrElement> numerator(fib.n);
        for (uint64_t c = 0; c < fib.bin.constraintsInfoDebug.size(); ++c) {
            const ParserParams &p = fib.bin.constraintsInfoDebug[c];
            expressions.calculateConstraint(c, trace, v, numerator.data());
            for (uint64_t i = p.firstRow; i < p.lastRow; ++i) assert(F().isZero(numerator[i]));
        }
    }

    // Q has at most qDeg·N + (qDeg + 1)·|O|max + 1 coefficients
    // (pilfflonk/docs/protocol.md#degrees), and here, with no blinding,
    // deg Q <= (qDeg + 1)(N − 1) − N < qDeg·N.
    const std::vector<FrElement> q = qCoefficients(fib, expressions, fib.stage1);
    uint64_t bound = 0; // Q's degree in the layout: that bound, with |O|max = 2 ({0, 1})
    for (const auto &f : fib.info.layout) {
        if (f.stage == fib.info.qStage()) bound = f.degree;
    }
    assert(bound == fib.info.qDeg * fib.n + (fib.info.qDeg + 1) * 2 + 1);
    assert(firstNonZeroFrom(q, fib.info.qDeg * fib.n) == q.size());
    assert(firstNonZeroFrom(q, bound) == q.size());
    assert(firstNonZeroFrom(q, 0) < fib.n);

    // At ξ it is the oracle's Q(ξ).
    const FrElement xi = fr(fib.oracle["xi"]);
    FrElement at = F().zero();
    for (uint64_t j = q.size(); j-- > 0;) {
        F().mul(at, at, xi);
        F().add(at, at, q[j]);
    }
    assert(eq(at, fr(fib.oracle["q"])));

    // The same with one thread: the blocks give the same bits whatever the team.
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    const std::vector<FrElement> single = qCoefficients(fib, expressions, fib.stage1);
    omp_set_num_threads(threads);
    for (uint64_t j = 0; j < q.size(); ++j) assert(eq(q[j], single[j]));

    // A mutated cell: the transition constraints fail on H, and Q is no polynomial of that degree.
    std::vector<std::vector<FrElement>> mutated = fib.stage1;
    F().add(mutated[0][100], mutated[0][100], F().one());
    const std::vector<FrElement> qm = qCoefficients(fib, expressions, mutated);
    assert(firstNonZeroFrom(qm, bound) < qm.size());
}

// The verifier mode: the qVerifier at ξ, from the oracle's evaluations, is the oracle's Q(ξ).
void testTheFibonaccisQVerifierIsTheOracles() {
    const Fibonacci fib;
    const ExpressionsBin qVerifier = ExpressionsBin::load(fixture("fibonacci/Fibonacci.qverifier.bin"));
    const Expressions expressions(qVerifier, fib.info);
    const FrElement xi = fr(fib.oracle["xi"]);

    PointValues point;
    point.zerofiers = PilFflonk::zerofiersAt(fib.info.nBits, fib.info.boundaries, xi);
    expectValues(point.zerofiers, fib.oracle["zerofiers"], "Zi at ξ");
    point.evals = frs(fib.oracle["evals"]);
    point.publics = frs(fib.oracle["publics"]);
    point.challenges = frs(fib.oracle["challenges"]);
    assert(point.evals.size() == fib.info.evMap.size());
    assert(eq(expressions.evaluateExpressionAt(fib.info.cExpId, point), fr(fib.oracle["q"])));

    // Another evaluation, another Q(ξ).
    F().add(point.evals[0], point.evals[0], F().one());
    assert(!eq(expressions.evaluateExpressionAt(fib.info.cExpId, point), fr(fib.oracle["q"])));
}

} // namespace

void runExpressionsTests() {
    testReadsTheRustSample();
    testRefusesWhatIsNotRevision3();
    testSampleGivesTheRustEvaluatorsValues();
    testZerofiersOnTheCoset();
    testTheFibonaccisQIsAPolynomialOfItsDegree();
    testTheFibonaccisQVerifierIsTheOracles();
}

} // namespace PilFflonkTest
