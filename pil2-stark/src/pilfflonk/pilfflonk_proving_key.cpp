#include "pilfflonk_proving_key.hpp"

#include <fcntl.h>
#include <omp.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <future>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

#include <optional>
#include <nlohmann/json.hpp>

#include "pilfflonk_error.hpp"
#include "pilfflonk_fr.hpp"
#include "pilfflonk_json.hpp"
#include "pilfflonk_transcript.hpp"
#include "timer.hpp"
#ifdef __USE_CUDA__
#include "pilfflonk_key_gpu.hpp"
#endif

namespace PilFflonk {

namespace {

using json = nlohmann::json;

// The files of the provingKey/ (pilfflonk/docs/formats.md#provingkey).
const char *const GLOBAL_INFO_FILE = "pilout.globalInfo.json";
const char *const BACKEND_DIR = "pilfflonk";
const char *const SRS_FILE = "pilfflonk.srs.bin";
const char *const SHIFT_SUMS_FILE = "pilfflonk.shift.bin";

// ---------------------------------------------------------------------------------------------
// pilout.globalInfo.json
// ---------------------------------------------------------------------------------------------

// The globalInfo's values, and its errors: "globalInfo: <where>: <what>".
constexpr JsonReader globalInfo("globalInfo");

const json &field(const json &object, const char *key, const std::string &where) {
    if (!object.is_object() || !object.contains(key)) {
        globalInfo.fail(where.empty() ? key : where, where.empty() ? "is missing" : std::string("has no ") + key);
    }
    return object[key];
}

// A path component taken from a file: a name, not a path.
std::string pathComponent(const json &value, const std::string &where) {
    const std::string s = globalInfo.str(value, where);
    if (s.empty() || s == "." || s == ".." || s.find('/') != std::string::npos || s.find('\0') != std::string::npos) {
        globalInfo.fail(where, "\"" + s + "\" is not a file name");
    }
    return s;
}

// The bytes of a file.
struct FileBytes {
    std::unique_ptr<uint8_t[]> data;
    uint64_t size = 0;
};

// The bytes of a .coefs as an AirKey takes them, its own.
AirKey::ConstantsBytes ownedConstants(FileBytes file) {
    AirKey::ConstantsBytes bytes;
    bytes.data = file.data.get();
    bytes.size = file.size;
    bytes.owner = std::move(file.data);
    return bytes;
}

// A file's bytes, read by every thread at once, a chunk each: a .const holds N rows of every fixed
// column, GBs at large N, which one thread copies from the page cache at a fraction of the speed of
// several, and the buffer is not cleared before (its pages are first touched by the threads that
// fill them).
// The bytes of the file at path past its first `skip`, which it must have.
FileBytes readBytes(const std::string &path, const char *what, uint64_t skip = 0) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw IoError(std::string(what) + ": cannot open " + path);
    }
    struct stat status;
    if (::fstat(fd, &status) != 0) {
        ::close(fd);
        throw IoError(std::string(what) + ": cannot read " + path);
    }
    if (static_cast<uint64_t>(status.st_size) < skip) {
        ::close(fd);
        throw FormatError(std::string(what) + ": " + path + " is shorter than its header");
    }
    FileBytes bytes;
    bytes.size = static_cast<uint64_t>(status.st_size) - skip;
    bytes.data.reset(new uint8_t[bytes.size]);
    constexpr uint64_t CHUNK = uint64_t(1) << 23;
    const uint64_t nChunks = (bytes.size + CHUNK - 1) / CHUNK;
    uint8_t *out = bytes.data.get();
    bool failed = false;
#pragma omp parallel for schedule(dynamic) reduction(|| : failed)
    for (uint64_t c = 0; c < nChunks; ++c) {
        const uint64_t end = std::min(bytes.size, (c + 1) * CHUNK);
        for (uint64_t at = c * CHUNK; at < end && !failed;) {
            const ssize_t n = ::pread(fd, out + at, end - at, static_cast<off_t>(skip + at));
            if (n > 0) {
                at += static_cast<uint64_t>(n);
            } else if (n == 0 || errno != EINTR) {
                // An error, or the end of a file that shrank since fstat.
                failed = true;
            }
        }
    }
    ::close(fd);
    if (failed) {
        throw IoError(std::string(what) + ": cannot read " + path);
    }
    return bytes;
}

const char COEFS_MAGIC[9] = "PFFCOEF1";

// An <air>.coefs: its coefficients, and the digest of its header.
struct Coefs {
    FileBytes bytes;
    KeyDigest digest;
};

Coefs readCoefs(const std::string &path) {
    uint8_t header[PRECOMPUTED_HEADER_BYTES];
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw IoError("coefs: cannot open " + path);
    }
    const ssize_t got = ::pread(fd, header, sizeof(header), 0);
    ::close(fd);
    if (got != static_cast<ssize_t>(sizeof(header))) {
        throw FormatError("coefs: " + path + " is shorter than its header");
    }
    const KeyDigest digest = precomputedDigest(header, COEFS_MAGIC, path);
    return Coefs{readBytes(path, "coefs", PRECOMPUTED_HEADER_BYTES), digest};
}

// ---------------------------------------------------------------------------------------------
// <air>.const and <air>.coefs
// ---------------------------------------------------------------------------------------------

// The nCols columns of n rows of `bytes`, row-major canonical little-endian scalars
// (pilfflonk/docs/formats.md#fixed-columns), into column-major Montgomery form: out[c·n + i] is
// column c at row i. Throws FormatError, naming the first value not below r.
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
        globalInfo.fail("the file", "must be an object");
    }
    if (globalInfo.str(field(j, "backend", ""), "backend") != "pilfflonk") {
        globalInfo.fail("backend", "must be \"pilfflonk\": this is not a pilfflonk provingKey/");
    }
    if (globalInfo.u64(field(j, "formatVersion", ""), "formatVersion") != 1) {
        globalInfo.fail("formatVersion", "must be 1, the version this prover reads");
    }
    if (globalInfo.str(field(j, "field", ""), "field") != "bn128") {
        globalInfo.fail("field", "must be \"bn128\"");
    }

    GlobalInfo info;
    info.name = pathComponent(field(j, "name", ""), "name");
    const json &airGroups = globalInfo.array(field(j, "air_groups", ""), "air_groups");
    for (size_t ag = 0; ag < airGroups.size(); ++ag) {
        info.airGroups.push_back(pathComponent(airGroups[ag], "air_groups[" + std::to_string(ag) + "]"));
    }
    const json &airs = globalInfo.array(field(j, "airs", ""), "airs");
    if (airs.size() != info.airGroups.size()) {
        globalInfo.fail("airs", "must have one list per airgroup");
    }
    for (size_t ag = 0; ag < airs.size(); ++ag) {
        const std::string where = "airs[" + std::to_string(ag) + "]";
        std::vector<Air> group;
        const json &list = globalInfo.array(airs[ag], where);
        for (size_t a = 0; a < list.size(); ++a) {
            const std::string at = where + "[" + std::to_string(a) + "]";
            group.push_back(Air{pathComponent(field(list[a], "name", at), at + ".name"),
                                globalInfo.u64(field(list[a], "num_rows", at), at + ".num_rows")});
        }
        info.airs.push_back(std::move(group));
    }
    info.nPublics = globalInfo.u64(field(j, "nPublics", ""), "nPublics");
    const json &proofValues = globalInfo.array(field(j, "proofValuesMap", ""), "proofValuesMap");
    for (size_t i = 0; i < proofValues.size(); ++i) {
        const std::string at = "proofValuesMap[" + std::to_string(i) + "]";
        info.proofValueStages.push_back(globalInfo.u64(field(proofValues[i], "stage", at), at + ".stage"));
    }
    return info;
}

GlobalInfo GlobalInfo::load(const std::string &path) {
    const std::string text = readText(path, "globalInfo: cannot open " + path, "globalInfo: cannot read " + path);
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

namespace {

// The fields of a hint of `kind` whose quotient gives its column: im_col's numerator and
// denominator, which calculateImHints reads, and gsum_col's and gprod_col's numerator_air and
// denominator_air, which calculateWitnessSTD does.
std::pair<const char *, const char *> quotientFields(StdHint::Kind kind) {
    return kind == StdHint::Kind::ImCol ? std::make_pair("numerator", "denominator")
                                        : std::make_pair("numerator_air", "denominator_air");
}

// The im_col, gsum_col and gprod_col hints of `bin`, checked against `info` as AirKey says, in the
// order AirKey::stdHints() has them. cmIds is AirKey::cmIds(). Throws FormatError, naming the AIR,
// the hint and what is wrong.
std::vector<StdHint> stdHintsOf(const ExpressionsBin &bin, const PilfflonkInfo &info,
                                const std::vector<std::vector<uint64_t>> &cmIds, const std::string &name) {
    std::vector<StdHint> hints;
    for (uint64_t h = 0; h < bin.hints.size(); ++h) {
        const Hint &hint = bin.hints[h];
        auto refused = [&](const std::string &what) {
            return FormatError(name + ": .bin: hint " + std::to_string(h) + " (" + hint.name + ") " + what);
        };
        StdHint entry;
        if (hint.name == "im_col") {
            entry.kind = StdHint::Kind::ImCol;
        } else if (hint.name == "gprod_col") {
            entry.kind = StdHint::Kind::Prod;
        } else if (hint.name == "gsum_col") {
            entry.kind = StdHint::Kind::Sum;
        } else if (hint.name == "im_airval") {
            throw refused("gives an air value, and pilfflonk has none (pilfflonk/docs/README.md#scope)");
        } else {
            throw refused("is none this prover computes: im_col, gsum_col and gprod_col");
        }
        // The value of a field of one value, or null if the hint has no such field.
        auto single = [&](const std::string &field) -> const HintFieldValue * {
            const auto it = std::find_if(hint.fields.begin(), hint.fields.end(),
                                         [&](const HintField &f) { return f.name == field; });
            if (it == hint.fields.end()) {
                return nullptr;
            }
            if (it->values.size() != 1 || !it->values[0].pos.empty()) {
                throw refused("holds an array in its field " + field + ", and the std's holds one value");
            }
            return &it->values[0];
        };
        auto required = [&](const std::string &field) -> const HintFieldValue & {
            const HintFieldValue *v = single(field);
            if (v == nullptr) {
                throw refused("has no field " + field);
            }
            return *v;
        };
        auto offsetOf = [&](const HintFieldValue &v, const std::string &field) {
            if (v.rowOffsetIndex >= info.openingPoints.size()) {
                throw refused("reads in its field " + field + " a column at opening point " +
                              std::to_string(v.rowOffsetIndex) + ", of " + std::to_string(info.openingPoints.size()));
            }
            return info.openingPoints[v.rowOffsetIndex];
        };

        entry.hint = h;
        entry.name = hint.name;
        // reference: a column of stage 2 or above, at its own row, not an im pol.
        const HintFieldValue &reference = required("reference");
        if (reference.op != HintOp::Cm || reference.id >= info.cmPolsMap.size()) {
            throw refused("has a reference that is not a committed column");
        }
        const PolMapEntry &column = info.cmPolsMap[reference.id];
        if (column.stage < 2 || column.stage > info.nStages || column.imPol || offsetOf(reference, "reference") != 0) {
            throw refused("has as reference " + column.name + " (stage " + std::to_string(column.stage) +
                          "), which is not a column of stages 2 … nStages, read at its own row, that is not an im pol");
        }
        entry.stage = column.stage;
        entry.stagePos = column.stagePos;
        entry.cmId = reference.id;

        // The numerator and the denominator: what addHintField takes. What they read, the order of the
        // hints checks below.
        auto input = [&](const std::string &field) {
            const HintFieldValue &v = required(field);
            HintInput in;
            if (v.op == HintOp::Cm || v.op == HintOp::Const) {
                const bool cm = v.op == HintOp::Cm;
                if (v.id >= (cm ? info.cmPolsMap.size() : info.constPolsMap.size())) {
                    throw refused("has in its field " + field + " a column its maps do not have");
                }
                in.kind = HintInput::Kind::Column;
                const PolMapEntry *p = cm ? &info.cmPolsMap[v.id] : nullptr;
                in.column = cm ? ColumnRead{uint32_t(p->stage), uint32_t(p->stagePos)} : ColumnRead{0, uint32_t(v.id)};
                in.offset = offsetOf(v, field);
            } else if (v.op == HintOp::Tmp) {
                in.kind = HintInput::Kind::Expression;
                in.expId = v.id;
            } else if (v.op == HintOp::Number) {
                in.kind = HintInput::Kind::Number;
                in.number = v.value;
            } else if (v.op == HintOp::AirValue) {
                throw refused("reads an air value in its field " + field +
                              ", and pilfflonk has none (pilfflonk/docs/README.md#scope)");
            } else {
                throw refused("has in its field " + field + " a value that is no expression, column or number");
            }
            return in;
        };
        const auto [numerator, denominator] = quotientFields(entry.kind);
        entry.numerator = input(numerator);
        entry.denominator = input(denominator);
        // result updates an airgroup value (updateAirgroupValue), and v1 has none: the STARK's
        // calculateWitnessSTD reads it, and numerator_direct and denominator_direct, only if the AIR has
        // airgroup values; the std writes a number in STD_MODE_ONE_INSTANCE. im_col has none of them.
        const HintFieldValue *result = entry.kind == StdHint::Kind::ImCol ? nullptr : single("result");
        if (!info.airgroupValuesMap.empty() || (result != nullptr && result->op != HintOp::Number)) {
            throw refused("updates an airgroup value, and pilfflonk has none (pilfflonk/docs/README.md#scope): the "
                          "std has one unless it is in STD_MODE_ONE_INSTANCE");
        }
        hints.push_back(std::move(entry));
    }

    // The order of the STARK: calculateImHints, and calculateWitnessSTD for the products and then the sums
    // (gen_proof.hpp), each by getHintIdsByName, in the .bin's order. calculateImHints computes nothing in
    // an AIR with no gsum_col or gprod_col.
    std::stable_sort(hints.begin(), hints.end(), [](const StdHint &a, const StdHint &b) { return a.kind < b.kind; });
    if (!hints.empty() && hints.back().kind == StdHint::Kind::ImCol) {
        failAir(name, ".bin: hint " + std::to_string(hints.front().hint) +
                          " (im_col) is of an AIR with no gsum_col or gprod_col, where the STARK's calculateImHints "
                          "computes none");
    }

    // What each hint reads is computed before it: the fixed columns, the columns of the stages before
    // its own, and of its own those of the hints before it in that order.
    for (uint64_t s = 2; s <= info.nStages; ++s) {
        std::vector<bool> computed(cmIds[s].size(), false);
        auto computedBefore = [&](const ColumnRead &c) {
            if (c.type == 0) {
                return c.index < info.constPolsMap.size();
            }
            if (c.type < s) {
                return c.index < cmIds[c.type].size() && cmIds[c.type][c.index] != AirKey::NOT_COMMITTED;
            }
            return c.type == s && c.index < computed.size() && computed[c.index];
        };
        for (const StdHint &hint : hints) {
            if (hint.stage != s) {
                continue;
            }
            const auto [numeratorField, denominatorField] = quotientFields(hint.kind);
            for (const auto &[field, in] :
                 {std::make_pair(numeratorField, &hint.numerator), std::make_pair(denominatorField, &hint.denominator)}) {
                std::vector<ColumnRead> reads;
                if (in->kind == HintInput::Kind::Column) {
                    reads.push_back(in->column);
                } else if (in->kind == HintInput::Kind::Expression) {
                    reads = columnsRead(bin, in->expId);
                }
                for (const ColumnRead &c : reads) {
                    if (!computedBefore(c)) {
                        failAir(name, ".bin: hint " + std::to_string(hint.hint) + " (" + hint.name +
                                          ") reads in its field " + field + " the column of stage " +
                                          std::to_string(c.type) + " at stagePos " + std::to_string(c.index) +
                                          ", which is not computed before it (of stage " + std::to_string(s) +
                                          ", the prover computes the columns of the im_col hints in their order, then "
                                          "the gprod_col ones and the gsum_col ones, and the im pols last)");
                    }
                }
            }
            computed[hint.stagePos] = true;
        }
    }

    // Each column of stages 2 … nStages but the im pols, the reference of one hint.
    std::vector<uint64_t> produced(info.cmPolsMap.size(), 0);
    for (const StdHint &hint : hints) {
        ++produced[hint.cmId];
    }
    for (uint64_t id = 0; id < info.cmPolsMap.size(); ++id) {
        const PolMapEntry &p = info.cmPolsMap[id];
        if (p.stage >= 2 && p.stage <= info.nStages && !p.imPol && produced[id] != 1) {
            failAir(name, ".bin: " + std::to_string(produced[id]) + " hints give the column " + p.name + " of stage " +
                              std::to_string(p.stage) + ", which one im_col, gsum_col or gprod_col must");
        }
    }
    return hints;
}

} // namespace

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
                          " points, and BN128's roots of unity allow at most 2^28 "
                          "(pilfflonk/docs/protocol.md#degrees)");
    }

    // The pieces of Q. maxQDegree < qDeg when it splits Q, so (m − 1)·qStride < qDeg·N < qCoefficients:
    // no product overflows, and the last piece keeps N + 1 coefficients at least.
    const uint64_t M = info.maxQDegree;
    if (M == 0 || M >= info.qDeg) {
        if (M != 0) {
            throw FormatError(name + ": maxQDegree = " + std::to_string(M) + " does not split Q of qDeg = " +
                              std::to_string(info.qDeg) + ", and then it is 0 (pilfflonk/docs/protocol.md#q-pieces)");
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

QPieceRange qPieceRange(const AirDegrees &d, uint64_t i) {
    const uint64_t start = i * d.qStride;
    return QPieceRange{start, i + 1 < d.qPieceCoefficients.size() ? d.qStride : d.qCoefficients - start};
}

AirKey::AirKey(PilfflonkInfo _info, ExpressionsBin _bin, const uint8_t *constants, uint64_t constantsBytes,
               const std::string &name, GpuKey *gpu)
    : AirKey(
          std::move(_info), std::move(_bin),
          [constants, constantsBytes] { return ConstantsBytes{constants, constantsBytes}; }, name, gpu) {}

AirKey::AirKey(PilfflonkInfo _info, ExpressionsBin _bin, const ConstantsSource &constants, const std::string &name,
               GpuKey *gpu)
    : AirKey(std::move(_info), std::move(_bin), name) {
    loadFixed(constants, gpu);
}

std::unique_ptr<AirKey> AirKey::withoutFixedColumns(PilfflonkInfo info, ExpressionsBin bin, const std::string &name) {
    return std::unique_ptr<AirKey>(new AirKey(std::move(info), std::move(bin), name));
}

AirKey::AirKey(PilfflonkInfo _info, ExpressionsBin _bin, const std::string &name)
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
        // one (pilfflonk/docs/protocol.md#degrees), and so k·coefficients once packed (pack()'s
        // max_j(k·deg p_j + j) + 1).
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

    // The pieces of Q (pilfflonk/docs/protocol.md#q-pieces): Q0 … Q<m−1> at stagePos (and
    // stageId) 0 … m − 1 of its stage, each packed once (the positions above), in f opened at ξ
    // whose degree is max_j(k·c_j + j) of their pieces' bounds c_j
    // (pilfflonk/docs/protocol.md#degrees), the one the setup gives them.
    const std::vector<uint64_t> &pieceBounds = airDegrees_.qPieceCoefficients;
    const std::vector<uint64_t> &pieces = cmIdsByStage[info.qStage()];
    if (pieces.size() != pieceBounds.size()) {
        fail("cmPolsMap has " + std::to_string(pieces.size()) + " pieces of Q, and Q is made of " +
             std::to_string(pieceBounds.size()) + " (pilfflonk/docs/protocol.md#q-pieces)");
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
                 " with a degree of " + std::to_string(entry.degree) + ", not its bound, " + std::to_string(degree) +
                 " (pilfflonk/docs/protocol.md#degrees)");
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
    hints = stdHintsOf(expressionsBin, info, cmIdsByStage, name);
    qColumns = readsOf(info.cExpId, "Q (cExpId)");
    for (const ColumnRead &c : qColumns) {
        const bool committed =
            exists(c) && c.type <= info.nStages &&
            (c.type == 0 ? constPositions[c.index].f : cmPositions[cmIdsByStage[c.type][c.index]].f) != NOT_COMMITTED;
        if (!committed) {
            fail("the code of Q reads " + columnName(c) + ", which the layout does not commit");
        }
    }

}

void AirKey::loadFixed(const ConstantsSource &constants, GpuKey *gpu) {
    const uint64_t N = n(), nConstants = pilfflonkInfo.nConstants;
#ifdef __USE_CUDA__
    // On the device first what does not need the fixed columns, while the .coefs may still be read.
    if (gpu != nullptr) {
        deviceKey = std::make_unique<GpuAirKey>(*gpu, *this);
    }
#endif
    ConstantsBytes bytes = constants();
    if (nConstants > std::numeric_limits<uint64_t>::max() / FR_BYTES / N || bytes.size != nConstants * N * FR_BYTES) {
        failAir(airName, ".coefs has " + std::to_string(bytes.size) + " bytes, and " + std::to_string(nConstants) +
                             " fixed columns of " + std::to_string(N) + " coefficients have " +
                             std::to_string(nConstants * N * FR_BYTES));
    }
    const uint64_t total = nConstants * N;
    const uint64_t first = firstNonCanonicalFr(bytes.data, total);
    if (first < total) {
        failAir(airName, ".coefs: the coefficient " + std::to_string(first % N) + " of the fixed column " +
                             std::to_string(first / N) + " is not below r");
    }
#ifdef __USE_CUDA__
    if (deviceKey != nullptr) {
        // Into Montgomery form on the device, where they stay: the host keeps nothing of them, and
        // frees its 2 GB-scale copy off the load's path.
        deviceKey->loadFixed(bytes.data);
        if (bytes.owner) {
            std::thread([owned = std::move(bytes.owner)] {}).detach();
        }
        return;
    }
#else
    (void)gpu;
#endif
    // Decoded in place, into the key's own bytes.
    std::unique_ptr<uint8_t[]> owned = std::move(bytes.owner);
    if (!owned) {
        owned.reset(new uint8_t[bytes.size]);
        std::memcpy(owned.get(), bytes.data, bytes.size);
    }
    FrElement *coefs = reinterpret_cast<FrElement *>(owned.get());
#pragma omp parallel for
    for (uint64_t i = 0; i < total; ++i) {
        coefs[i] = fromCanonicalFr(owned.get() + i * FR_BYTES);
    }
    fixedPolys.reserve(nConstants);
    for (uint64_t c = 0; c < nConstants; ++c) {
        fixedPolys.emplace_back(Poly::fromReservedBuffer(AltBn128::Engine::engine, coefs + c * N, N));
    }
    fixedCoefs = std::move(owned);
}

AirKey::~AirKey() = default;

const FrElement *AirKey::fixedEvaluations(uint64_t c) const {
    // Built aside and kept only once whole: a call that throws leaves nothing, and the next one tries
    // again.
    std::call_once(fixedEvaluated, [this] {
        const uint64_t N = n(), nConstants = pilfflonkInfo.nConstants;
        if (nConstants == 0) {
            return;
        }
        std::unique_ptr<FrElement[]> evaluations(new FrElement[nConstants * N]);
#ifdef __USE_CUDA__
        if (deviceKey != nullptr) {
            deviceKey->fixedToHost(evaluations.get());
        }
#endif
        if (fixedCoefs) {
            std::memcpy(evaluations.get(), fixedCoefs.get(), nConstants * N * sizeof(FrElement));
        }
        std::vector<FrElement *> columns(nConstants);
        for (uint64_t k = 0; k < nConstants; ++k) {
            columns[k] = evaluations.get() + k * N;
        }
        extension->ntt(columns.data(), columns.data(), nConstants);
        fixedEvals = std::move(evaluations);
    });
    return fixedEvals.get() + c * n();
}

Poly *AirKey::fixedPolynomial(uint64_t c) const {
#ifdef __USE_CUDA__
    if (deviceKey != nullptr) {
        // As fixedEvaluations: kept only once whole.
        std::call_once(fixedCopied, [this] {
            const uint64_t N = n(), nConstants = pilfflonkInfo.nConstants;
            std::unique_ptr<uint8_t[]> coefs(new uint8_t[nConstants * N * sizeof(FrElement)]);
            FrElement *elements = reinterpret_cast<FrElement *>(coefs.get());
            deviceKey->fixedToHost(elements);
            std::vector<std::unique_ptr<Poly>> polys;
            polys.reserve(nConstants);
            for (uint64_t k = 0; k < nConstants; ++k) {
                polys.push_back(mirrorPolynomial(elements + k * N, N, deviceKey->fixedCount(k),
                                                 airName + ": the fixed column " + pilfflonkInfo.constPolsMap[k].name));
            }
            fixedCoefs = std::move(coefs);
            fixedPolys = std::move(polys);
        });
    }
#endif
    return fixedPolys[c].get();
}

ShplonkComponent AirKey::fixedComponent(uint64_t c) const {
#ifdef __USE_CUDA__
    if (deviceKey != nullptr) {
        return ShplonkComponent::elsewhere(n(), deviceKey->fixedDegree(c));
    }
#endif
    return fixedPolynomial(c);
}

std::vector<G1Point> AirKey::fixedCommitments(const Srs &srs) const {
#ifdef __USE_CUDA__
    if (deviceKey != nullptr && deviceKey->gpuKey().holds(srs)) {
        return deviceKey->fixedCommitments();
    }
#endif
    TimerStart(PILFFLONK_FIXED_COMMITMENTS);
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
    TimerStopAndLog(PILFFLONK_FIXED_COMMITMENTS);
    return commitments;
}

uint64_t AirKey::blindLength(uint64_t f) const {
    const LayoutEntry &entry = pilfflonkInfo.layout.at(f);
    return entry.stage >= 1 && entry.stage <= pilfflonkInfo.nStages ? entry.offsets.size() + 1 : 0;
}

uint64_t AirKey::wMsm() const {
    uint64_t w = 1;
    for (const LayoutEntry &entry : pilfflonkInfo.layout) {
        const uint64_t roots = entry.k * entry.offsets.size();
        if (entry.degree > roots) {
            w = std::max(w, entry.degree - roots);
        }
    }
    return w;
}

uint64_t AirKey::wpMsm() const {
    uint64_t longest = 1;
    for (const LayoutEntry &entry : pilfflonkInfo.layout) {
        longest = std::max(longest, entry.degree);
    }
    return std::max<uint64_t>(longest - 1, 1);
}

std::vector<uint64_t> AirKey::msmLengths() const {
    std::vector<uint64_t> lengths;
    for (const LayoutEntry &entry : pilfflonkInfo.layout) {
        lengths.push_back(entry.degree);
    }
    lengths.push_back(wMsm());
    lengths.push_back(wpMsm());
    return lengths;
}

std::vector<uint8_t> fixedCoefficientsOf(const uint8_t *constants, uint64_t bytes, uint64_t nBits, uint64_t nConstants,
                                         const std::string &name) {
    const uint64_t N = uint64_t(1) << nBits, total = N * nConstants;
    if (bytes != total * FR_BYTES) {
        throw FormatError(name + ": " + std::to_string(bytes) + " bytes, and " + std::to_string(nConstants) +
                          " fixed columns of " + std::to_string(N) + " rows have " + std::to_string(total * FR_BYTES));
    }
    std::vector<uint8_t> out(total * FR_BYTES);
    if (total == 0) {
        return out;
    }
    std::unique_ptr<FrElement[]> evals(new FrElement[total]), coefs(new FrElement[total]);
    decodeColumns(constants, N, nConstants, evals.get(), name);
    std::vector<FrElement *> e(nConstants), c(nConstants);
    for (uint64_t k = 0; k < nConstants; ++k) {
        e[k] = evals.get() + k * N;
        c[k] = coefs.get() + k * N;
    }
    Lde(nBits, nBits).intt(e.data(), c.data(), nConstants);
#pragma omp parallel for
    for (uint64_t i = 0; i < total; ++i) {
        FrElement canonical;
        AltBn128::Engine::engine.fr.fromMontgomery(canonical, coefs[i]);
        std::memcpy(out.data() + i * FR_BYTES, &canonical, FR_BYTES);
    }
    return out;
}

std::unique_ptr<AirKey> AirKey::load(const std::string &dir, const std::string &name, GpuKey *gpu) {
    const std::string base = dir + "/" + name;
    PilfflonkInfo info = PilfflonkInfo::load(base + ".pilfflonkinfo.json");
    ExpressionsBin bin = ExpressionsBin::load(base + ".bin");
    FileBytes constants = readCoefs(base + ".coefs").bytes;
    const AirKey::ConstantsSource source = [&constants] { return ownedConstants(std::move(constants)); };
    return std::make_unique<AirKey>(std::move(info), std::move(bin), source, name, gpu);
}

// ---------------------------------------------------------------------------------------------
// ProvingKey
// ---------------------------------------------------------------------------------------------

void checkSrsFits(const AirKey &air, uint64_t nG1) {
    const std::vector<LayoutEntry> &layout = air.info().layout;
    for (uint64_t f = 0; f < layout.size(); ++f) {
        if (layout[f].degree > nG1) {
            throw FormatError(air.name() + ": layout f" + std::to_string(f) + " has " +
                              std::to_string(layout[f].degree) + " coefficients, and the SRS " + std::to_string(nG1) +
                              " powers [τ^i]₁");
        }
    }
}

ProvingKey::ProvingKey(GlobalInfo _info, Srs _srs, std::vector<std::vector<std::unique_ptr<AirKey>>> _airs,
                       std::shared_ptr<const GpuKey> _gpu)
    : info(std::move(_info)), gpu(std::move(_gpu)), structuredReferenceString(std::move(_srs)),
      airKeys(std::move(_airs)) {
#ifdef __USE_CUDA__
    if (gpu && gpu->nPowers() != structuredReferenceString.nG1()) {
        throw std::invalid_argument("provingKey: its GPU holds " + std::to_string(gpu->nPowers()) +
                                    " points, and its SRS " + std::to_string(structuredReferenceString.nG1()) +
                                    " powers [τ^i]₁");
    }
#endif
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
#ifdef __USE_CUDA__
            const GpuKey *onGpu = key.device() != nullptr ? &key.device()->gpuKey() : nullptr;
#else
            const GpuKey *onGpu = nullptr;
#endif
            if (onGpu != gpu.get()) {
                throw std::invalid_argument("provingKey: " + key.name() +
                                            (gpu ? "'s key is not on the provingKey's GPU" : "'s key is on a GPU"));
            }
            const PilfflonkInfo &air = key.info();
            const GlobalInfo::Air &expected = info.airs[ag][a];
            if (air.airgroupId != ag || air.airId != a || air.name != expected.name || key.n() != expected.numRows) {
                throw FormatError(key.name() + ": its pilfflonkinfo is not air " + std::to_string(a) + " of airgroup " +
                                  std::to_string(ag) + " of the globalInfo, " + expected.name + " of " +
                                  std::to_string(expected.numRows) + " rows");
            }
            checkSrsFits(key, structuredReferenceString.nG1());
        }
    }
}

namespace {

// Throws std::invalid_argument, naming `function`, if there is no GPU (gpuAvailable()).
void requireGpu(const char *function) {
    if (!gpuAvailable()) {
#ifdef __USE_CUDA__
        throw std::invalid_argument(std::string(function) +
                                    ": no GPU: CUDA sees no device of compute capability 7.0 or above (or no driver)");
#else
        throw std::invalid_argument(std::string(function) +
                                    ": no GPU: this library was built without it (provers/starks-lib-c/build.rs found "
                                    "no nvcc, or the feature cpu-only)");
#endif
    }
}

// An AIR of a provingKey/: its airgroup, its name, and its files' path without their extensions.
struct AirFiles {
    uint64_t airgroup;
    std::string name;
    std::string base;
};

// Every AIR of the globalInfo `info` of the provingKey/ at `dir`, in canonical order.
std::vector<AirFiles> airFilesOf(const std::string &dir, const GlobalInfo &info) {
    std::vector<AirFiles> airs;
    for (uint64_t ag = 0; ag < info.airs.size(); ++ag) {
        for (const GlobalInfo::Air &air : info.airs[ag]) {
            const std::string airDir = dir + "/" + info.name + "/" + info.airGroups[ag] + "/airs/" + air.name + "/air";
            airs.push_back(AirFiles{ag, air.name, airDir + "/" + air.name});
        }
    }
    return airs;
}

std::string srsPathOf(const std::string &dir, const GlobalInfo &info) {
    return dir + "/" + info.name + "/" + BACKEND_DIR + "/" + SRS_FILE;
}

std::string shiftSumsPathOf(const std::string &dir, const GlobalInfo &info) {
    return dir + "/" + info.name + "/" + BACKEND_DIR + "/" + SHIFT_SUMS_FILE;
}

} // namespace

std::unique_ptr<ProvingKey> ProvingKey::load(const std::string &dir, Device device, const GpuKeyOptions &options) {
    if (device == Device::Gpu) {
        // CUDA's initialisation, on its first call: what is left of it if a warm-up started it
        // before (pilfflonk/docs/performance.md#the-start-of-a-proof).
        TimerStart(PILFFLONK_GPU_INIT);
        const bool available = gpuAvailable();
        TimerStopAndLog(PILFFLONK_GPU_INIT);
        if (!available) {
            requireGpu("ProvingKey::load");
        }
    }
    GlobalInfo info = GlobalInfo::load(dir + "/" + GLOBAL_INFO_FILE);
    const std::vector<AirFiles> files = airFilesOf(dir, info);
    // Each AIR's .coefs on a thread of its own: the first one's from now, while the SRS is read and
    // copied to the device, and each next one's from when the AIR before it reads its own, while that
    // AIR's key is built (pilfflonk/docs/performance.md#loading-a-key-on-the-gpu).
    auto readConstants = [&files](uint64_t i) {
        const std::string path = files[i].base + ".coefs";
        return std::async(std::launch::async, [path] { return readCoefs(path); });
    };
    std::future<Coefs> pending;
    if (!files.empty()) {
        pending = readConstants(0);
    }
    std::optional<KeyDigest> digest;
    TimerStart(PILFFLONK_LOAD_SRS);
    // Its points checked only on demand (pilfflonk_ctx_check_srs): the setup made it from a checked ptau.
    Srs srs = Srs::load(srsPathOf(dir, info), false);
    TimerStopAndLog(PILFFLONK_LOAD_SRS);
    std::shared_ptr<GpuKey> gpu;
#ifdef __USE_CUDA__
    if (device == Device::Gpu) {
        TimerStart(PILFFLONK_GPU_SRS);
        gpu = std::make_shared<GpuKey>(srs, options);
        KeyDigest shiftDigest;
        gpu->addShiftSums(readShiftSums(shiftSumsPathOf(dir, info), &shiftDigest));
        digest = shiftDigest;
        TimerStopAndLog(PILFFLONK_GPU_SRS);
    }
#else
    (void)options;
#endif
    TimerStart(PILFFLONK_LOAD_AIRS);
    std::vector<std::vector<std::unique_ptr<AirKey>>> airs(info.airs.size());
    for (uint64_t i = 0; i < files.size(); ++i) {
        PilfflonkInfo airInfo = PilfflonkInfo::load(files[i].base + ".pilfflonkinfo.json");
        ExpressionsBin bin = ExpressionsBin::load(files[i].base + ".bin");
        const AirKey::ConstantsSource source = [&] {
            Coefs constants = pending.get();
            if (i + 1 < files.size()) {
                pending = readConstants(i + 1);
            }
            // Every precomputed file of one setup's: one digest.
            if (!digest) {
                digest = constants.digest;
            } else if (*digest != constants.digest) {
                throw FormatError(files[i].base + ".coefs and the key's other precomputed files (the shift sums or "
                                                  "another AIR's .coefs) were made for different vkeys: rerun setup-pilfflonk");
            }
            return ownedConstants(std::move(constants.bytes));
        };
        airs[files[i].airgroup].push_back(
            std::make_unique<AirKey>(std::move(airInfo), std::move(bin), source, files[i].name, gpu.get()));
    }
    TimerStopAndLog(PILFFLONK_LOAD_AIRS);
#ifdef __USE_CUDA__
    if (gpu) {
        logCopies(gpu->copies(), "KEY", CopyVolume::Totals());
    }
#endif
    auto key = std::make_unique<ProvingKey>(std::move(info), std::move(srs), std::move(airs), std::move(gpu));
    key->precomputed = digest.value_or(KeyDigest{});
    return key;
}

#ifdef __USE_CUDA__
void AirKey::setExec(const pilfflonk_exec_static &exec) const { deviceKey->setExec(exec); }
#endif

void ProvingKey::setExec(uint64_t airgroupId, uint64_t airId, const pilfflonk_exec_static &exec) const {
#ifdef __USE_CUDA__
    if (air(airgroupId, airId).device() != nullptr) {
        air(airgroupId, airId).setExec(exec);
        return;
    }
#endif
    (void)airgroupId;
    (void)airId;
    (void)exec;
    throw std::invalid_argument("ProvingKey::setExec: the exec's parts are for a key on the GPU");
}

void ProvingKey::snapshotDevice() const {
#ifdef __USE_CUDA__
    if (gpu) {
        gpu->snapshot();
    }
#endif
}

void ProvingKey::restoreDevice() const {
#ifdef __USE_CUDA__
    if (gpu) {
        gpu->restore();
    }
#endif
}

void ProvingKey::precompute(const std::string &dir, const KeyDigest &digest) {
    const GlobalInfo info = GlobalInfo::load(dir + "/" + GLOBAL_INFO_FILE);
    std::vector<uint64_t> lengths;
    for (const AirFiles &air : airFilesOf(dir, info)) {
        const std::unique_ptr<AirKey> key = AirKey::withoutFixedColumns(
            PilfflonkInfo::load(air.base + ".pilfflonkinfo.json"), ExpressionsBin::load(air.base + ".bin"), air.name);
        const std::vector<uint64_t> own = key->msmLengths();
        lengths.insert(lengths.end(), own.begin(), own.end());
        const FileBytes constants = readBytes(air.base + ".const", "const");
        const std::vector<uint8_t> coefs = fixedCoefficientsOf(constants.data.get(), constants.size, key->info().nBits,
                                                               key->info().nConstants, air.name + ".const");
        const std::string path = air.base + ".coefs";
        const auto header = precomputedHeader(COEFS_MAGIC, digest);
        FILE *file = std::fopen(path.c_str(), "wb");
        const bool written = file != nullptr &&
                             std::fwrite(header.data(), 1, header.size(), file) == header.size() &&
                             std::fwrite(coefs.data(), 1, coefs.size(), file) == coefs.size();
        if (file == nullptr || std::fclose(file) != 0 || !written) {
            throw IoError(path + ": cannot write the fixed coefficients: " + std::strerror(errno));
        }
    }
    const std::string srsPath = srsPathOf(dir, info);
    writeShiftSums(shiftSumsPathOf(dir, info), shiftSums(Srs::load(srsPath), lengths), digest);
}

DeviceBytes ProvingKey::requiredDeviceBytes(const std::string &dir) {
    requireGpu("ProvingKey::requiredDeviceBytes");
#ifdef __USE_CUDA__
    const GlobalInfo info = GlobalInfo::load(dir + "/" + GLOBAL_INFO_FILE);
    const uint64_t nG1 = Srs::powersIn(srsPathOf(dir, info));
    std::vector<std::unique_ptr<AirKey>> keys;
    std::vector<const AirKey *> airs;
    for (const AirFiles &air : airFilesOf(dir, info)) {
        keys.push_back(AirKey::withoutFixedColumns(PilfflonkInfo::load(air.base + ".pilfflonkinfo.json"),
                                                   ExpressionsBin::load(air.base + ".bin"), air.name));
        airs.push_back(keys.back().get());
    }
    return deviceBytesOf(nG1, airs, gpuMultiprocessors());
#else
    (void)dir;
    return DeviceBytes();
#endif
}

uint64_t ProvingKey::freeDeviceBytes() {
    requireGpu("ProvingKey::freeDeviceBytes");
#ifdef __USE_CUDA__
    return gpuFreeBytes();
#else
    return 0;
#endif
}

const AirKey &ProvingKey::air(uint64_t airgroupId, uint64_t airId) const {
    if (airgroupId >= airKeys.size() || airId >= airKeys[airgroupId].size()) {
        throw std::invalid_argument("the provingKey has no air " + std::to_string(airId) + " in airgroup " +
                                    std::to_string(airgroupId));
    }
    return *airKeys[airgroupId][airId];
}

} // namespace PilFflonk
