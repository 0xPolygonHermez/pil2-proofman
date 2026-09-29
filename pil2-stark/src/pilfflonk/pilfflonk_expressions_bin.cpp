#include "pilfflonk_expressions_bin.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "pilfflonk_error.hpp"
#include "pilfflonk_fr.hpp"

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

// Reads a part of the file, never past its end: what BinFile's reads do not ensure within a
// section, and its readString does not for a string without its NUL.
class Reader {
public:
    Reader(const uint8_t *data, uint64_t size, const std::string &name, const char *what)
        : data(data), size(size), name(name), what(what) {}

    [[noreturn]] void fail(const std::string &message) const {
        throw FormatError(name + ": " + what + ": " + message);
    }

    const uint8_t *take(uint64_t n) {
        if (n > size - pos) {
            fail("it ends before its content does");
        }
        const uint8_t *bytes = data + pos;
        pos += n;
        return bytes;
    }

    uint32_t u32() {
        const uint8_t *b = take(4);
        return uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
    }

    uint64_t u64() {
        const uint64_t low = u32();
        return low | uint64_t(u32()) << 32;
    }

    std::string string() {
        const uint8_t *begin = data + pos;
        const uint8_t *end = static_cast<const uint8_t *>(std::memchr(begin, 0, size - pos));
        if (end == nullptr) {
            fail("a string has no end");
        }
        pos += (end - begin) + 1;
        return std::string(reinterpret_cast<const char *>(begin), end - begin);
    }

    void finish() const {
        if (pos != size) {
            fail("it has " + std::to_string(size - pos) + " bytes after its content");
        }
    }

private:
    const uint8_t *data;
    uint64_t size;
    uint64_t pos = 0;
    const std::string &name;
    const char *what;
};

std::string hex(uint32_t value) {
    static const char *digits = "0123456789abcdef";
    std::string out = "0x";
    bool started = false;
    for (int shift = 28; shift >= 0; shift -= 4) {
        const uint32_t digit = (value >> shift) & 0xf;
        if (digit != 0 || started || shift == 0) {
            out += digits[digit];
            started = true;
        }
    }
    return out;
}

// The STARK's rank of a source's kind (operations_map_value of setup/pil2-stark/src/io/parser_args.rs,
// for dimension 1): an op's first source never ranks above its second.
uint32_t rank(const OperandTypes &t, uint32_t type) {
    if (t.isColumn(type) || type == t.zi()) return 0;
    if (type == t.tmp()) return 1;
    if (type == t.publics()) return 2;
    if (type == t.numbers()) return 3;
    if (type == t.airValues()) return 4;
    if (type == t.proofValues()) return 5;
    if (type == t.airgroupValues()) return 9;
    if (type == t.challenges()) return 11;
    return 12; // evals
}

// A section's code after its entries: the ops, the args and the numbers, and every entry's code
// checked against them, as the Rust reader does.
void readCode(Reader &r, const OperandTypes &types, std::vector<ParserParams> &entries, uint32_t nOps, uint32_t nArgs,
              uint32_t nNumbers, ParserArgs &out, const char *what) {
    uint64_t opsOffset = 0;
    uint64_t argsOffset = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        const ParserParams &e = entries[i];
        if (e.opsOffset != opsOffset || e.argsOffset != argsOffset) {
            r.fail("entry " + std::to_string(i) + ": its ops or args do not follow the previous entry's");
        }
        if (uint64_t(e.nArgs) != uint64_t(e.nOps) * ARGS_PER_OP) {
            r.fail("entry " + std::to_string(i) + ": " + std::to_string(e.nArgs) + " args for " +
                   std::to_string(e.nOps) + " ops");
        }
        opsOffset += e.nOps;
        argsOffset += e.nArgs;
    }
    if (opsOffset != nOps || argsOffset != nArgs) {
        r.fail("the entries have " + std::to_string(opsOffset) + " ops and " + std::to_string(argsOffset) +
               " args, not " + std::to_string(nOps) + " and " + std::to_string(nArgs));
    }

    const uint8_t *ops = r.take(nOps);
    out.ops.assign(ops, ops + nOps);
    for (uint32_t i = 0; i < nOps; ++i) {
        if (out.ops[i] != 0) {
            r.fail("op " + std::to_string(i) + " is of dimensions " + std::to_string(out.ops[i]) +
                   ", and every value has dimension 1");
        }
    }
    out.args.resize(nArgs);
    for (uint32_t i = 0; i < nArgs; ++i) {
        out.args[i] = r.u32();
    }
    Engine::Fr &fr = Engine::engine.fr;
    out.numbers.resize(nNumbers);
    for (uint32_t i = 0; i < nNumbers; ++i) {
        const uint8_t *bytes = r.take(FR_BYTES);
        if (!isCanonicalFr(bytes)) {
            r.fail("number " + std::to_string(i) + " is not below r");
        }
        fr.fromRprLE(out.numbers[i], bytes, FR_BYTES);
    }
    r.finish();

    std::vector<bool> used(nNumbers, false);
    for (size_t i = 0; i < entries.size(); ++i) {
        const ParserParams &e = entries[i];
        const std::string entry = std::string(what) + ", entry " + std::to_string(i);
        if (e.nOps == 0) {
            r.fail(entry + ": a code block with no ops");
        }
        if (e.nTemp > e.nOps) {
            r.fail(entry + ": " + std::to_string(e.nTemp) + " temporaries for " + std::to_string(e.nOps) + " ops");
        }
        std::vector<bool> written(e.nTemp, false);
        for (uint32_t k = 0; k < e.nOps; ++k) {
            const uint32_t *op = &out.args[e.argsOffset + uint64_t(k) * ARGS_PER_OP];
            const std::string at = entry + ", op " + std::to_string(k);
            if (op[0] > OP_SUB_SWAP) {
                r.fail(at + ": unknown opType " + std::to_string(op[0]));
            }
            for (int s = 0; s < 2; ++s) {
                const uint32_t type = op[2 + 3 * s];
                const uint32_t arg1 = op[3 + 3 * s];
                const uint32_t arg2 = op[4 + 3 * s];
                const std::string source = "(" + std::to_string(type) + ", " + std::to_string(arg1) + ", " +
                                           std::to_string(arg2) + ")";
                const bool known = types.isColumn(type) || (type == types.zi() && arg1 >= 1) ||
                                   type == types.tmp() || (type >= types.publics() && type <= types.evals());
                if (!known) {
                    r.fail(at + ": " + source + " is no operand of an AIR of " + std::to_string(types.nStages) +
                           " stages");
                }
                if (!types.isColumn(type) && arg2 != 0) {
                    r.fail(at + ": an operand of type " + std::to_string(type) + " with arg2 " + std::to_string(arg2));
                }
                if (type == types.tmp() && (arg1 >= e.nTemp || !written[arg1])) {
                    r.fail(at + ": temporary " + std::to_string(arg1) +
                           " is read before it is written, or is not below nTemp");
                }
                if (type == types.numbers()) {
                    if (arg1 >= nNumbers) {
                        r.fail(at + ": number " + std::to_string(arg1) + ", of " + std::to_string(nNumbers));
                    }
                    used[arg1] = true;
                }
            }
            if (rank(types, op[2]) > rank(types, op[5])) {
                r.fail(at + ": its sources are not in the STARK's order");
            }
            if (op[1] >= e.nTemp) {
                r.fail(at + ": temporary " + std::to_string(op[1]) + " is not below nTemp");
            }
            written[op[1]] = true;
        }
        const uint32_t lastDest = out.args[e.argsOffset + uint64_t(e.nOps - 1) * ARGS_PER_OP + 1];
        if (e.destId != lastDest) {
            r.fail(entry + ": destId " + std::to_string(e.destId) + " is not the last op's dest, " +
                   std::to_string(lastDest));
        }
    }
    for (uint32_t i = 0; i < nNumbers; ++i) {
        if (!used[i]) {
            r.fail("number " + std::to_string(i) + " is not used");
        }
    }
}

// nTemp, nOps, opsOffset, nArgs and argsOffset, in the order both sections have them.
void readCodeFields(Reader &r, ParserParams &p) {
    p.nTemp = r.u32();
    p.nOps = r.u32();
    p.opsOffset = r.u32();
    p.nArgs = r.u32();
    p.argsOffset = r.u32();
}

} // namespace

ExpressionsBin ExpressionsBin::load(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw IoError(path + ": cannot open the file");
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad()) {
        throw IoError(path + ": cannot read the file");
    }
    return parse(bytes.data(), bytes.size(), path);
}

ExpressionsBin ExpressionsBin::parse(const uint8_t *data, uint64_t size, const std::string &name) {
    if (data == nullptr && size != 0) {
        throw std::invalid_argument("ExpressionsBin::parse: data is null");
    }
    ExpressionsBin bin;

    // The container: "chps", pilfflonk's version, and sections 1, 2 and 3 in order.
    Reader file(data, size, name, "the file");
    if (std::memcmp(file.take(4), "chps", 4) != 0) {
        file.fail("it is not a \"chps\" binfile");
    }
    const uint32_t version = file.u32();
    if (version != EXPRESSIONS_BIN_VERSION) {
        file.fail("version " + hex(version) + ", and pilfflonk's is " + hex(EXPRESSIONS_BIN_VERSION) +
                  ": not a pilfflonk bytecode of revision 2");
    }
    if (file.u32() != EXPRESSIONS_BIN_N_SECTIONS) {
        file.fail("it does not have 3 sections");
    }
    const uint8_t *sections[EXPRESSIONS_BIN_N_SECTIONS];
    uint64_t sizes[EXPRESSIONS_BIN_N_SECTIONS];
    for (uint32_t id = 1; id <= EXPRESSIONS_BIN_N_SECTIONS; ++id) {
        const uint32_t sectionId = file.u32();
        if (sectionId != id) {
            file.fail("section " + std::to_string(sectionId) + " where section " + std::to_string(id) + " goes");
        }
        sizes[id - 1] = file.u64();
        sections[id - 1] = file.take(sizes[id - 1]);
    }
    file.finish();

    // Section 1: the prefix, the counts and the entries.
    Reader r1(sections[0], sizes[0], name, "the expressions");
    const uint32_t prefixVersion = r1.u32();
    const uint32_t n8 = r1.u32();
    const uint8_t *modulus = r1.take(FR_BYTES);
    if (prefixVersion != EXPRESSIONS_BIN_VERSION || n8 != FR_BYTES ||
        std::memcmp(modulus, FR_MODULUS_LE, FR_BYTES) != 0) {
        r1.fail("section 1 starts with version " + hex(prefixVersion) + " and n8 " + std::to_string(n8) +
                ", not those of a revision-2 bytecode over BN254");
    }
    bin.nStages = r1.u32();
    if (bin.nStages > EXPRESSIONS_BIN_MAX_N_STAGES) {
        r1.fail(std::to_string(bin.nStages) + " stages");
    }
    bin.maxTmp = r1.u32();
    bin.maxArgs = r1.u32();
    bin.maxOps = r1.u32();
    const uint32_t nOps = r1.u32();
    const uint32_t nArgs = r1.u32();
    const uint32_t nNumbers = r1.u32();
    const uint32_t nExpressions = r1.u32();
    std::vector<ParserParams> expressions;
    for (uint32_t i = 0; i < nExpressions; ++i) {
        ParserParams p;
        p.expId = r1.u32();
        p.destId = r1.u32();
        p.stage = r1.u32();
        readCodeFields(r1, p);
        p.line = r1.string();
        expressions.push_back(std::move(p));
    }
    readCode(r1, bin.types(), expressions, nOps, nArgs, nNumbers, bin.expressionsBinArgsExpressions, "the expressions");
    for (ParserParams &p : expressions) {
        const uint64_t expId = p.expId;
        if (!bin.expressionsInfo.emplace(expId, std::move(p)).second) {
            r1.fail("expression " + std::to_string(expId) + " is there twice");
        }
    }

    // Section 2.
    Reader r2(sections[1], sizes[1], name, "the constraints");
    const uint32_t nOpsDebug = r2.u32();
    const uint32_t nArgsDebug = r2.u32();
    const uint32_t nNumbersDebug = r2.u32();
    const uint32_t nConstraints = r2.u32();
    for (uint32_t i = 0; i < nConstraints; ++i) {
        ParserParams p;
        p.stage = r2.u32();
        p.destId = r2.u32();
        p.firstRow = r2.u32();
        p.lastRow = r2.u32();
        readCodeFields(r2, p);
        const uint32_t imPol = r2.u32();
        p.line = r2.string();
        if (p.firstRow > p.lastRow || imPol > 1) {
            r2.fail("constraint " + std::to_string(i) + ": rows " + std::to_string(p.firstRow) + ".." +
                    std::to_string(p.lastRow) + ", imPol " + std::to_string(imPol));
        }
        p.imPol = imPol == 1;
        bin.constraintsInfoDebug.push_back(std::move(p));
    }
    readCode(r2, bin.types(), bin.constraintsInfoDebug, nOpsDebug, nArgsDebug, nNumbersDebug,
             bin.expressionsBinArgsConstraints, "the constraints");

    // The maxima cover both sections, as the STARK's.
    uint32_t maxTmp = 0, maxArgs = 0, maxOps = 0;
    auto cover = [&](const ParserParams &p) {
        maxTmp = std::max(maxTmp, p.nTemp);
        maxArgs = std::max(maxArgs, p.nArgs);
        maxOps = std::max(maxOps, p.nOps);
    };
    for (const auto &entry : bin.expressionsInfo) cover(entry.second);
    for (const ParserParams &p : bin.constraintsInfoDebug) cover(p);
    if (maxTmp != bin.maxTmp || maxArgs != bin.maxArgs || maxOps != bin.maxOps) {
        r1.fail("maxTmp, maxArgs and maxOps are " + std::to_string(bin.maxTmp) + ", " + std::to_string(bin.maxArgs) +
                " and " + std::to_string(bin.maxOps) + ", and the code's are " + std::to_string(maxTmp) + ", " +
                std::to_string(maxArgs) + " and " + std::to_string(maxOps));
    }

    // Section 3: no hints in revision 2.
    Reader r3(sections[2], sizes[2], name, "the hints");
    const uint32_t nHints = r3.u32();
    r3.finish();
    if (nHints != 0) {
        r3.fail(std::to_string(nHints) + " hints: revision 2 has none");
    }
    return bin;
}

const ParserParams &ExpressionsBin::expression(uint64_t expId) const {
    auto it = expressionsInfo.find(expId);
    if (it == expressionsInfo.end()) {
        throw std::invalid_argument("ExpressionsBin: no expression " + std::to_string(expId));
    }
    return it->second;
}

} // namespace PilFflonk
