#ifndef PILFFLONK_EXPRESSIONS_BIN_HPP
#define PILFFLONK_EXPRESSIONS_BIN_HPP

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "alt_bn128.hpp"

namespace PilFflonk {

using FrElement = AltBn128::Engine::FrElement;

// <air>.bin, revision 3 (spec A.6): the prover's bytecode over Fr, and its hints. The Rust crate
// pilfflonk-setup writes it (setup/pilfflonk/src/bytecode.rs, which documents the format field by
// field) and has the reference reader. The format is the STARK's prover .bin, the one
// pil2-stark/src/starkpil/expressions/expressions_bin.cpp reads, with every value of dimension 1:
//
// - no destDim, nTemp3 or maxTmp3, and no dim of a hint's expression;
// - args of 32 bits, not 16;
// - numbers of 32 bytes, canonical Fr little-endian, not u64, in the code and in the hints;
// - section 1 starts with a prefix: the version again, n8 = 32, r and nStages.
//
// This reader is as strict as the Rust one: whatever the file breaks, it throws FormatError, naming
// the file and what is wrong; IoError if the file cannot be read. It never exits.

constexpr uint32_t EXPRESSIONS_BIN_VERSION = 0x70660003; // "pf" and revision 3
constexpr uint32_t EXPRESSIONS_BIN_EXPRESSIONS_SECTION = 1;
constexpr uint32_t EXPRESSIONS_BIN_CONSTRAINTS_SECTION = 2;
constexpr uint32_t EXPRESSIONS_BIN_HINTS_SECTION = 3;
constexpr uint32_t EXPRESSIONS_BIN_N_SECTIONS = 3;
// An op: opType dest aType aArg1 aArg2 bType bArg1 bArg2, as in the STARK.
constexpr uint32_t ARGS_PER_OP = 8;

// The STARK's operation codes, args[0] of an op.
enum OpCode : uint32_t { OP_ADD = 0, OP_SUB = 1, OP_MUL = 2, OP_SUB_SWAP = 3 };

// The operand types, (type, arg1, arg2), of an AIR of nStages stages: the STARK's buffer indices,
// with no custom commits (P5).
//
// | type | operand | arg1 | arg2 |
// | 0 | fixed column | its id (column of <air>.const) | openingPoints index |
// | 1 … nStages + 1 | committed column of that stage | stagePos | openingPoints index |
// | nStages + 2 | Zi | 1 + boundary | 0 |
// | tmp() | temporary | its id | 0 |
// | tmp() + 2 … tmp() + 8 | public, number, air value, proof value, airgroup value, challenge, evaluation | index | 0 |
//
// tmp() + 1 (the STARK's tmp3), nStages + 3 (xDivXSubXi) and Zi with arg1 0 (x) are never written.
// As in the STARK, a source of type > tmp() + 1 is the same on every point.
struct OperandTypes {
    uint32_t nStages;

    bool isColumn(uint32_t type) const { return type <= nStages + 1; }
    uint32_t zi() const { return nStages + 2; }
    uint32_t tmp() const { return nStages + 4; }
    uint32_t publics() const { return tmp() + 2; }
    uint32_t numbers() const { return tmp() + 3; }
    uint32_t airValues() const { return tmp() + 4; }
    uint32_t proofValues() const { return tmp() + 5; }
    uint32_t airgroupValues() const { return tmp() + 6; }
    uint32_t challenges() const { return tmp() + 7; }
    uint32_t evals() const { return tmp() + 8; }
};

// The most stages an AIR can have for its operand types, up to tmp() + 8, to fit in 32 bits.
constexpr uint32_t EXPRESSIONS_BIN_MAX_N_STAGES = UINT32_MAX - 12;

// An expression (section 1) or a constraint (section 2): the STARK's ParserParams without destDim
// and nTemp3. expId is an expression's; firstRow, lastRow and imPol a constraint's.
struct ParserParams {
    uint32_t stage = 0;
    uint32_t expId = 0;
    uint32_t nTemp = 0;
    uint32_t nOps = 0;
    uint32_t opsOffset = 0;
    uint32_t nArgs = 0;
    uint32_t argsOffset = 0;
    uint32_t firstRow = 0; // the rows it holds on: firstRow <= i < lastRow
    uint32_t lastRow = 0;
    uint32_t destId = 0; // the temporary the value is in after the last op
    bool imPol = false;
    std::string line;
};

// A section's code, as the STARK's ParserArgs: every entry's ops and args one after another, and
// the numbers they index. numbers are in ffiasm's Montgomery form.
struct ParserArgs {
    std::vector<uint8_t> ops;
    std::vector<uint32_t> args;
    std::vector<FrElement> numbers;
};

// The kind of a value of a hint field: the STARK's opType names of them (opType2string), but custom
// (P5).
enum class HintOp { Cm, Const, Tmp, Number, String, Public, Challenge, AirValue, AirgroupValue, ProofValue };

// A value of a hint field (section 3): the STARK's HintFieldValue, with its ids. id is the cmPolsMap
// index of a Cm, the constPolsMap index of a Const, the expId of a Tmp (an expression of section 1,
// which the reader checks) and the index in their maps of the others'; the reader checks the rest of
// them against nothing, as it has no pilfflonkinfo.
struct HintFieldValue {
    HintOp op = HintOp::Number;
    uint64_t id = 0;             // all but Number and String
    uint64_t rowOffsetIndex = 0; // Cm and Const: the index of its row offset in openingPoints
    FrElement value;             // Number, in Montgomery form
    std::string stringValue;     // String
    std::vector<uint64_t> pos;   // its position in the field's array: empty for a single value
};

struct HintField {
    std::string name;
    std::vector<HintFieldValue> values;
};

struct Hint {
    std::string name;
    std::vector<HintField> fields;
};

class ExpressionsBin {
public:
    // Reads the file at path.
    static ExpressionsBin load(const std::string &path);

    // The same from the file's bytes; `name` is what the errors call it.
    static ExpressionsBin parse(const uint8_t *data, uint64_t size, const std::string &name);

    OperandTypes types() const { return OperandTypes{nStages}; }

    // The expression of expId; throws std::invalid_argument if there is none.
    const ParserParams &expression(uint64_t expId) const;

    // The indices in hints of the hints named `name`, in their order: the STARK's getHintIdsByName.
    std::vector<uint64_t> hintIds(const std::string &name) const;

    uint32_t nStages = 0;
    // The largest nTemp, nArgs and nOps of both sections, as the STARK's.
    uint32_t maxTmp = 0;
    uint32_t maxArgs = 0;
    uint32_t maxOps = 0;
    std::map<uint64_t, ParserParams> expressionsInfo; // by expId, each once
    std::vector<ParserParams> constraintsInfoDebug;   // in the pilout's order
    ParserArgs expressionsBinArgsExpressions;
    ParserArgs expressionsBinArgsConstraints;
    std::vector<Hint> hints; // section 3, in the file's order
};

} // namespace PilFflonk

#endif
