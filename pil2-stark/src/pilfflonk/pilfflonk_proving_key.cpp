#include "pilfflonk_proving_key.hpp"

#include <omp.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "pilfflonk_error.hpp"
#include "pilfflonk_fr.hpp"
#include "pilfflonk_transcript.hpp"

namespace PilFflonk {

namespace {

using json = nlohmann::json;

// The files of spec §4.2.6.
const char *const GLOBAL_INFO_FILE = "pilout.globalInfo.json";
const char *const BACKEND_DIR = "pilfflonk";
const char *const SRS_FILE = "pilfflonk.srs.bin";

// ---------------------------------------------------------------------------------------------
// pilout.globalInfo.json
// ---------------------------------------------------------------------------------------------

[[noreturn]] void failGlobalInfo(const std::string &where, const std::string &what) {
    throw FormatError("globalInfo: " + where + ": " + what);
}

const json &field(const json &object, const char *key, const std::string &where) {
    if (!object.is_object() || !object.contains(key)) {
        failGlobalInfo(where.empty() ? key : where, where.empty() ? "is missing" : std::string("has no ") + key);
    }
    return object[key];
}

uint64_t u64(const json &value, const std::string &where) {
    if (!value.is_number_unsigned()) {
        failGlobalInfo(where, "must be an unsigned integer");
    }
    return value.get<uint64_t>();
}

std::string str(const json &value, const std::string &where) {
    if (!value.is_string()) {
        failGlobalInfo(where, "must be a string");
    }
    return value.get<std::string>();
}

const json &array(const json &value, const std::string &where) {
    if (!value.is_array()) {
        failGlobalInfo(where, "must be an array");
    }
    return value;
}

// A path component taken from a file: a name, not a path.
std::string pathComponent(const json &value, const std::string &where) {
    const std::string s = str(value, where);
    if (s.empty() || s == "." || s == ".." || s.find('/') != std::string::npos || s.find('\0') != std::string::npos) {
        failGlobalInfo(where, "\"" + s + "\" is not a file name");
    }
    return s;
}

std::string readText(const std::string &path, const char *what) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw IoError(std::string(what) + ": cannot open " + path);
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad()) {
        throw IoError(std::string(what) + ": cannot read " + path);
    }
    return text;
}

std::vector<uint8_t> readBytes(const std::string &path, const char *what) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw IoError(std::string(what) + ": cannot open " + path);
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad()) {
        throw IoError(std::string(what) + ": cannot read " + path);
    }
    return bytes;
}

// ---------------------------------------------------------------------------------------------
// <air>.const
// ---------------------------------------------------------------------------------------------

// The nCols columns of n rows of `bytes`, row-major canonical little-endian scalars (spec A.6), into
// column-major Montgomery form: out[c·n + i] is column c at row i. Throws FormatError, naming the
// first value not below r.
void decodeColumns(const uint8_t *bytes, uint64_t n, uint64_t nCols, FrElement *out, const std::string &name) {
    const uint64_t total = n * nCols;
    const uint64_t first = firstNonCanonicalFr(bytes, total);
    if (first < total) {
        throw FormatError(name + ": the value of row " + std::to_string(first / nCols) + ", column " +
                          std::to_string(first % nCols) + " is not below r");
    }
#pragma omp parallel for
    for (uint64_t v = 0; v < total; ++v) {
        out[(v % nCols) * n + v / nCols] = fromCanonicalFr(bytes + v * FR_BYTES);
    }
}

[[noreturn]] void failAir(const std::string &name, const std::string &what) {
    throw FormatError(name + ": " + what);
}

uint64_t ceilLog2(uint64_t x) {
    uint64_t bits = 0;
    while (bits < 64 && (uint64_t(1) << bits) < x) {
        ++bits;
    }
    return bits;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// GlobalInfo
// ---------------------------------------------------------------------------------------------

GlobalInfo GlobalInfo::parse(const std::string &text) {
    json j;
    try {
        j = json::parse(text);
    } catch (const json::parse_error &e) {
        throw FormatError(std::string("globalInfo: not valid JSON: ") + e.what());
    }
    if (!j.is_object()) {
        failGlobalInfo("the file", "must be an object");
    }
    if (str(field(j, "backend", ""), "backend") != "pilfflonk") {
        failGlobalInfo("backend", "must be \"pilfflonk\": this is not a pilfflonk provingKey/");
    }
    if (u64(field(j, "formatVersion", ""), "formatVersion") != 1) {
        failGlobalInfo("formatVersion", "must be 1, the version this prover reads");
    }
    if (str(field(j, "field", ""), "field") != "bn254") {
        failGlobalInfo("field", "must be \"bn254\"");
    }

    GlobalInfo info;
    info.name = pathComponent(field(j, "name", ""), "name");
    const json &airGroups = array(field(j, "air_groups", ""), "air_groups");
    for (size_t ag = 0; ag < airGroups.size(); ++ag) {
        info.airGroups.push_back(pathComponent(airGroups[ag], "air_groups[" + std::to_string(ag) + "]"));
    }
    const json &airs = array(field(j, "airs", ""), "airs");
    if (airs.size() != info.airGroups.size()) {
        failGlobalInfo("airs", "must have one list per airgroup");
    }
    for (size_t ag = 0; ag < airs.size(); ++ag) {
        const std::string where = "airs[" + std::to_string(ag) + "]";
        std::vector<Air> group;
        const json &list = array(airs[ag], where);
        for (size_t a = 0; a < list.size(); ++a) {
            const std::string at = where + "[" + std::to_string(a) + "]";
            group.push_back(Air{pathComponent(field(list[a], "name", at), at + ".name"),
                                u64(field(list[a], "num_rows", at), at + ".num_rows")});
        }
        info.airs.push_back(std::move(group));
    }
    info.nPublics = u64(field(j, "nPublics", ""), "nPublics");
    const json &proofValues = array(field(j, "proofValuesMap", ""), "proofValuesMap");
    for (size_t i = 0; i < proofValues.size(); ++i) {
        const std::string at = "proofValuesMap[" + std::to_string(i) + "]";
        info.proofValueStages.push_back(u64(field(proofValues[i], "stage", at), at + ".stage"));
    }
    return info;
}

GlobalInfo GlobalInfo::load(const std::string &path) {
    const std::string text = readText(path, "globalInfo");
    try {
        return parse(text);
    } catch (const FormatError &e) {
        throw FormatError(path + ": " + e.what());
    }
}

// ---------------------------------------------------------------------------------------------
// AirKey
// ---------------------------------------------------------------------------------------------

std::vector<ColumnRead> columnsRead(const ExpressionsBin &bin, uint64_t expId) {
    const ParserParams &p = bin.expression(expId);
    const OperandTypes types = bin.types();
    std::vector<ColumnRead> reads;
    for (uint64_t k = 0; k < p.nOps; ++k) {
        const uint32_t *op = &bin.expressionsBinArgsExpressions.args[p.argsOffset + k * ARGS_PER_OP];
        // opType dest aType aArg1 aArg2 bType bArg1 bArg2
        for (const uint32_t *source : {op + 2, op + 5}) {
            if (types.isColumn(source[0])) {
                reads.push_back(ColumnRead{source[0], source[1]});
            }
        }
    }
    std::sort(reads.begin(), reads.end());
    reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
    return reads;
}

AirDegrees airDegrees(const PilfflonkInfo &info, const std::string &name) {
    AirDegrees d{};
    if (info.nBits > MAX_NBITS_EXT) {
        throw FormatError(name + ": nBits = " + std::to_string(info.nBits) + " exceeds 28");
    }
    d.n = uint64_t(1) << info.nBits;
    for (const LayoutEntry &f : info.layout) {
        if (f.stage >= 1 && f.stage <= info.nStages) {
            d.maxOpenings = std::max<uint64_t>(d.maxOpenings, f.offsets.size());
        }
    }
    // qDeg·N + (qDeg+1)·|O|_max + 1 <= (2·qDeg + 1)·2^28 + 1, with N <= 2^28 and |O|_max <= N: below
    // 2^63 if qDeg < 2^34. A larger qDeg has an extended domain above 2^28 anyway.
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    if (info.qDeg > (max >> (MAX_NBITS_EXT + 2)) || d.maxOpenings > d.n) {
        throw FormatError(name + ": qDeg = " + std::to_string(info.qDeg) + " and |O|max = " +
                          std::to_string(d.maxOpenings) + " give a Q bound that does not fit in 64 bits");
    }
    d.qCoefficients = info.qDeg * d.n + (info.qDeg + 1) * d.maxOpenings + 1;
    d.nBitsExt = ceilLog2(std::max(d.qCoefficients, d.n + d.maxOpenings + 1));
    if (d.nBitsExt > MAX_NBITS_EXT) {
        throw FormatError(name + ": the extended domain has 2^" + std::to_string(d.nBitsExt) +
                          " points, and BN254's roots of unity allow at most 2^28 (spec A.1)");
    }

    // The pieces of Q. maxQDegree < qDeg when it splits Q, so (m − 1)·qStride < qDeg·N < qCoefficients:
    // no product overflows, and the last piece keeps N + 1 coefficients at least.
    const uint64_t M = info.maxQDegree;
    if (M == 0 || M >= info.qDeg) {
        if (M != 0) {
            throw FormatError(name + ": maxQDegree = " + std::to_string(M) + " does not split Q of qDeg = " +
                              std::to_string(info.qDeg) + ", and then it is 0 (spec A.1)");
        }
        d.qStride = 0;
        d.qPieceCoefficients = {d.qCoefficients};
        return d;
    }
    const uint64_t m = (info.qDeg + M - 1) / M;
    d.qStride = M * d.n;
    d.qPieceCoefficients.assign(m - 1, d.qStride + 2);
    d.qPieceCoefficients.push_back(d.qCoefficients - (m - 1) * d.qStride);
    return d;
}

AirKey::AirKey(PilfflonkInfo _info, ExpressionsBin _bin, const uint8_t *constants, uint64_t constantsBytes,
               const std::string &name)
    : airName(name), pilfflonkInfo(std::move(_info)), expressionsBin(std::move(_bin)) {
    const PilfflonkInfo &info = pilfflonkInfo;
    auto fail = [&](const std::string &what) { failAir(name, what); };

    interpreter = std::make_unique<Expressions>(expressionsBin, info);
    airDegrees_ = airDegrees(info, name);
    const uint64_t N = airDegrees_.n;
    extension = std::make_unique<Lde>(info.nBits, airDegrees_.nBitsExt);

    // The rows of each constraint of the .bin (section 2, which check runs) lie in the trace; its
    // reader checked firstRow <= lastRow.
    for (uint64_t c = 0; c < expressionsBin.constraintsInfoDebug.size(); ++c) {
        const ParserParams &p = expressionsBin.constraintsInfoDebug[c];
        if (p.lastRow > N) {
            fail(".bin: constraint " + std::to_string(c) + " holds on the rows " + std::to_string(p.firstRow) +
                 " <= i < " + std::to_string(p.lastRow) + ", and the trace has " + std::to_string(N));
        }
    }

    // The layout: each column committed once.
    constPositions.assign(info.constPolsMap.size(), LayoutPosition{NOT_COMMITTED, 0});
    cmPositions.assign(info.cmPolsMap.size(), LayoutPosition{NOT_COMMITTED, 0});
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        const LayoutEntry &entry = info.layout[f];
        std::vector<LayoutPosition> &positions = entry.stage == 0 ? constPositions : cmPositions;
        for (uint64_t j = 0; j < entry.pols.size(); ++j) {
            const uint64_t id = entry.pols[j].id;
            if (positions[id].f != NOT_COMMITTED) {
                fail("layout f" + std::to_string(f) + " packs " + (entry.stage == 0 ? "const " : "cm ") +
                     std::to_string(id) + ", which an earlier f packs too");
            }
            positions[id] = LayoutPosition{f, j};
        }
        if (entry.stage == info.qStage()) {
            continue; // below, with the pieces of Q
        }
        // Its k columns fit in its bound: N coefficients for a fixed one, N + |O| + 1 for a committed
        // one (spec A.2), and so k·coefficients once packed (pack()'s max_j(k·deg p_j + j) + 1).
        const uint64_t coefficients = N + blindLength(f);
        if (entry.k > entry.degree / coefficients) {
            fail("layout f" + std::to_string(f) + " has a degree of " + std::to_string(entry.degree) + ", below the " +
                 std::to_string(entry.k) + " polynomials of " + std::to_string(coefficients) +
                 " coefficients it packs");
        }
    }

    // The witness columns, by stageId.
    std::vector<uint64_t> byStageId;
    std::vector<bool> seen;
    for (const PolMapEntry &p : info.cmPolsMap) {
        if (p.stage != 1 || p.imPol) {
            continue;
        }
        if (p.stageId >= seen.size()) {
            seen.resize(p.stageId + 1, false);
            byStageId.resize(p.stageId + 1, 0);
        }
        if (seen[p.stageId]) {
            fail("two witness columns of stage 1 have stageId " + std::to_string(p.stageId));
        }
        seen[p.stageId] = true;
        byStageId[p.stageId] = p.stagePos;
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        fail("the stageIds of the witness columns of stage 1 are not 0 … C − 1");
    }
    witness = std::move(byStageId);

    // The columns by stage and stagePos (the reader checked every stagePos against mapSectionsN).
    cmIdsByStage.assign(info.qStage() + 1, {});
    for (uint64_t s = 1; s <= info.qStage(); ++s) {
        const auto width = info.mapSectionsN.find("cm" + std::to_string(s));
        cmIdsByStage[s].assign(width == info.mapSectionsN.end() ? 0 : width->second, NOT_COMMITTED);
    }
    for (uint64_t id = 0; id < info.cmPolsMap.size(); ++id) {
        const PolMapEntry &p = info.cmPolsMap[id];
        if (cmIdsByStage[p.stage][p.stagePos] != NOT_COMMITTED) {
            fail("cmPolsMap " + std::to_string(id) + " has the stagePos of an earlier column of stage " +
                 std::to_string(p.stage));
        }
        cmIdsByStage[p.stage][p.stagePos] = id;
    }
    nFixed = std::count_if(info.layout.begin(), info.layout.end(), [](const LayoutEntry &f) { return f.stage == 0; });

    // The pieces of Q (spec A.1): Q0 … Q<m−1> at stagePos (and stageId) 0 … m − 1 of its stage, each
    // packed once (the positions above), in f opened at ξ whose degree is A.2's cost max_j(k·c_j + j)
    // of their pieces' bounds c_j, the one the setup gives them.
    const std::vector<uint64_t> &pieceBounds = airDegrees_.qPieceCoefficients;
    const std::vector<uint64_t> &pieces = cmIdsByStage[info.qStage()];
    if (pieces.size() != pieceBounds.size()) {
        fail("cmPolsMap has " + std::to_string(pieces.size()) + " pieces of Q, and Q is made of " +
             std::to_string(pieceBounds.size()) + " (spec A.1)");
    }
    qPositions.assign(pieces.size(), LayoutPosition{NOT_COMMITTED, 0});
    for (uint64_t i = 0; i < pieces.size(); ++i) {
        const PolMapEntry &p = info.cmPolsMap[pieces[i]];
        if (p.name != "Q" + std::to_string(i) || p.stageId != i || p.imPol) {
            fail("cmPolsMap " + std::to_string(pieces[i]) + " (" + p.name + ") is at stagePos " + std::to_string(i) +
                 " of Q's stage: it must be piece Q" + std::to_string(i) + ", of stageId " + std::to_string(i));
        }
        qPositions[i] = cmPositions[pieces[i]];
        if (qPositions[i].f == NOT_COMMITTED) {
            fail("the layout does not pack piece Q" + std::to_string(i) + " of Q");
        }
    }
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        const LayoutEntry &entry = info.layout[f];
        if (entry.stage != info.qStage()) {
            continue;
        }
        if (entry.offsets != std::vector<int64_t>{0}) {
            fail("layout f" + std::to_string(f) + " holds Q, which is opened at ξ only");
        }
        uint64_t degree = 0;
        for (uint64_t j = 0; j < entry.k; ++j) {
            const uint64_t bound = pieceBounds[info.cmPolsMap[entry.pols[j].id].stagePos];
            if (bound > (std::numeric_limits<uint64_t>::max() - j) / entry.k) {
                fail("layout f" + std::to_string(f) + " packs " + std::to_string(entry.k) +
                     " pieces of Q, whose degree does not fit in 64 bits");
            }
            degree = std::max(degree, bound * entry.k + j);
        }
        if (entry.degree != degree) {
            fail("layout f" + std::to_string(f) + " holds " + (pieceBounds.size() > 1 ? "pieces of Q" : "Q") +
                 " with a degree of " + std::to_string(entry.degree) + ", not the bound of spec A.1, " +
                 std::to_string(degree));
        }
    }

    // What the code reads: the columns of the stages computed before it, and only committed ones for
    // Q, whose columns are extended from their committed polynomials.
    auto readsOf = [&](uint64_t expId, const std::string &what) {
        try {
            return columnsRead(expressionsBin, expId);
        } catch (const std::invalid_argument &) {
            failAir(name, ".bin has no code for " + what + ", expression " + std::to_string(expId));
        }
    };
    auto columnName = [&](const ColumnRead &c) {
        return c.type == 0 ? "const " + std::to_string(c.index)
                           : "the column of stage " + std::to_string(c.type) + " at stagePos " + std::to_string(c.index);
    };
    auto exists = [&](const ColumnRead &c) {
        return c.type == 0 ? c.index < info.constPolsMap.size()
                           : c.index < cmIdsByStage[c.type].size() && cmIdsByStage[c.type][c.index] != NOT_COMMITTED;
    };
    for (const PolMapEntry &p : info.cmPolsMap) {
        if (!p.imPol) {
            continue;
        }
        if (!p.hasExpId) {
            fail("im pol " + p.name + " has no expId");
        }
        for (const ColumnRead &c : readsOf(p.expId, "im pol " + p.name)) {
            if (c.type > p.stage || !exists(c)) {
                fail("the code of im pol " + p.name + " (stage " + std::to_string(p.stage) + ") reads " +
                     columnName(c) + ", which is not computed before it");
            }
        }
    }
    qColumns = readsOf(info.cExpId, "Q (cExpId)");
    for (const ColumnRead &c : qColumns) {
        const bool committed =
            exists(c) && c.type <= info.nStages &&
            (c.type == 0 ? constPositions[c.index].f : cmPositions[cmIdsByStage[c.type][c.index]].f) != NOT_COMMITTED;
        if (!committed) {
            fail("the code of Q reads " + columnName(c) + ", which the layout does not commit");
        }
    }

    // The fixed columns.
    const uint64_t nConstants = info.nConstants;
    if (nConstants > std::numeric_limits<uint64_t>::max() / FR_BYTES / N ||
        constantsBytes != nConstants * N * FR_BYTES) {
        fail(".const has " + std::to_string(constantsBytes) + " bytes, and " + std::to_string(nConstants) +
             " fixed columns of " + std::to_string(N) + " rows have " + std::to_string(nConstants * N * FR_BYTES));
    }
    if (nConstants > 0) {
        fixedEvals.reset(new FrElement[nConstants * N]);
        fixedCoefs.reset(new FrElement[nConstants * N]);
        decodeColumns(constants, N, nConstants, fixedEvals.get(), name + ".const");
        std::vector<FrElement *> evals(nConstants), coefs(nConstants);
        for (uint64_t c = 0; c < nConstants; ++c) {
            evals[c] = fixedEvals.get() + c * N;
            coefs[c] = fixedCoefs.get() + c * N;
        }
        fixedPolys = extension->intt(evals.data(), coefs.data(), nConstants);
    }
}

std::vector<G1Point> AirKey::fixedCommitments(const Srs &srs) const {
    std::vector<G1Point> commitments;
    commitments.reserve(nFixed);
    for (uint64_t f = 0; f < nFixed; ++f) {
        const LayoutEntry &entry = pilfflonkInfo.layout[f];
        std::vector<Poly *> components;
        components.reserve(entry.pols.size());
        for (const LayoutPol &pol : entry.pols) {
            components.push_back(fixedPolynomial(pol.id));
        }
        commitments.push_back(commitPacked(srs, components.data(), components.size()));
    }
    return commitments;
}

uint64_t AirKey::blindLength(uint64_t f) const {
    const LayoutEntry &entry = pilfflonkInfo.layout.at(f);
    return entry.stage >= 1 && entry.stage <= pilfflonkInfo.nStages ? entry.offsets.size() + 1 : 0;
}

std::unique_ptr<AirKey> AirKey::load(const std::string &dir, const std::string &name) {
    const std::string base = dir + "/" + name;
    PilfflonkInfo info = PilfflonkInfo::load(base + ".pilfflonkinfo.json");
    ExpressionsBin bin = ExpressionsBin::load(base + ".bin");
    const std::vector<uint8_t> constants = readBytes(base + ".const", "const");
    return std::make_unique<AirKey>(std::move(info), std::move(bin), constants.data(), constants.size(), name);
}

// ---------------------------------------------------------------------------------------------
// ProvingKey
// ---------------------------------------------------------------------------------------------

ProvingKey::ProvingKey(GlobalInfo _info, Srs _srs, std::vector<std::vector<std::unique_ptr<AirKey>>> _airs)
    : info(std::move(_info)), structuredReferenceString(std::move(_srs)), airKeys(std::move(_airs)) {
    if (airKeys.size() != info.airs.size()) {
        throw FormatError("provingKey: " + std::to_string(airKeys.size()) + " airgroups of keys for the " +
                          std::to_string(info.airs.size()) + " of the globalInfo");
    }
    for (uint64_t ag = 0; ag < airKeys.size(); ++ag) {
        if (airKeys[ag].size() != info.airs[ag].size()) {
            throw FormatError("provingKey: airgroup " + std::to_string(ag) + " has keys for " +
                              std::to_string(airKeys[ag].size()) + " AIRs, and the globalInfo has " +
                              std::to_string(info.airs[ag].size()));
        }
        for (uint64_t a = 0; a < airKeys[ag].size(); ++a) {
            const AirKey &key = *airKeys[ag][a];
            const PilfflonkInfo &air = key.info();
            const GlobalInfo::Air &expected = info.airs[ag][a];
            if (air.airgroupId != ag || air.airId != a || air.name != expected.name || key.n() != expected.numRows) {
                throw FormatError(key.name() + ": its pilfflonkinfo is not air " + std::to_string(a) + " of airgroup " +
                                  std::to_string(ag) + " of the globalInfo, " + expected.name + " of " +
                                  std::to_string(expected.numRows) + " rows");
            }
            for (uint64_t f = 0; f < air.layout.size(); ++f) {
                if (air.layout[f].degree > structuredReferenceString.nG1()) {
                    throw FormatError(key.name() + ": layout f" + std::to_string(f) + " has " +
                                      std::to_string(air.layout[f].degree) + " coefficients, and the SRS " +
                                      std::to_string(structuredReferenceString.nG1()) + " powers [τ^i]₁");
                }
            }
        }
    }
}

std::unique_ptr<ProvingKey> ProvingKey::load(const std::string &dir) {
    GlobalInfo info = GlobalInfo::load(dir + "/" + GLOBAL_INFO_FILE);
    Srs srs = Srs::load(dir + "/" + info.name + "/" + BACKEND_DIR + "/" + SRS_FILE);
    std::vector<std::vector<std::unique_ptr<AirKey>>> airs(info.airs.size());
    for (uint64_t ag = 0; ag < info.airs.size(); ++ag) {
        for (const GlobalInfo::Air &air : info.airs[ag]) {
            const std::string airDir = dir + "/" + info.name + "/" + info.airGroups[ag] + "/airs/" + air.name + "/air";
            airs[ag].push_back(AirKey::load(airDir, air.name));
        }
    }
    return std::make_unique<ProvingKey>(std::move(info), std::move(srs), std::move(airs));
}

const AirKey &ProvingKey::air(uint64_t airgroupId, uint64_t airId) const {
    if (airgroupId >= airKeys.size() || airId >= airKeys[airgroupId].size()) {
        throw std::invalid_argument("the provingKey has no air " + std::to_string(airId) + " in airgroup " +
                                    std::to_string(airgroupId));
    }
    return *airKeys[airgroupId][airId];
}

} // namespace PilFflonk
