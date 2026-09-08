// Interpreter tests: hand-assembled bytecode evaluated against arithmetic
// computed directly, so the whole path -- opcode dispatch, operand addressing,
// temporary routing -- is checked end to end rather than piecewise.
#include <gtest/gtest.h>
#include <vector>

#include "fr_evaluator.hpp"

using PilFflonk::ARGS_PER_OP;
using PilFflonk::ExpressionsLayout;
using PilFflonk::FrEl;
using PilFflonk::FrOp;
using PilFflonk::ParserArgs;
using PilFflonk::ParserParams;
using PilFflonk::StepsParams;

static AltBn128::Engine &E = AltBn128::Engine::engine;

namespace {

constexpr uint64_t DOMAIN = 8;
constexpr uint64_t N_CONST = 2;
constexpr uint64_t N_TRACE = 2;

ExpressionsLayout layout() {
    ExpressionsLayout l;
    l.nStages = 1;
    l.nConstants = N_CONST;
    l.mapSectionsN = {N_CONST, N_TRACE, 0};
    l.mapOffsets = {0, 0, 0};
    l.nextStrides = {0, 1};
    return l;
}

/// One operand descriptor: three words.
void operand(std::vector<uint16_t> &v, uint16_t type, uint16_t pos, uint16_t stride) {
    v.push_back(type);
    v.push_back(pos);
    v.push_back(stride);
}

std::vector<FrEl> grid(uint64_t rows, uint64_t cols, uint64_t bias) {
    std::vector<FrEl> g(rows * cols);
    for (uint64_t r = 0; r < rows; r++)
        for (uint64_t c = 0; c < cols; c++) g[r * cols + c] = E.fr.set(bias + r * 10 + c);
    return g;
}

uint64_t asUI(const FrEl &e) { return std::stoull(E.fr.toString(e)); }

} // namespace

// trace[row][0] + trace[row][1], the simplest complete program.
TEST(FR_EVALUATOR, evaluatesASingleOperation) {
    auto l = layout();
    auto trace = grid(DOMAIN, N_TRACE, 0);

    StepsParams params;
    params.trace = trace.data();

    std::vector<uint16_t> args;
    args.push_back((uint16_t)FrOp::add);
    args.push_back(0);              // dest temp (unused: single op)
    operand(args, 1, 0, 0);         // trace column 0
    operand(args, 1, 1, 0);         // trace column 1
    std::vector<uint8_t> ops = {0};

    ParserParams pp{1, 0, ARGS_PER_OP, 0};
    ParserArgs pa{ops.data(), args.data()};

    const uint64_t nrows = 4;
    std::vector<FrEl> tmp(4 * nrows), out(nrows);

    PilFflonk::evaluateExpression(l, params, pp, pa, tmp.data(), out.data(), 0, nrows, DOMAIN, false, false);

    for (uint64_t j = 0; j < nrows; j++) {
        EXPECT_EQ(asUI(out[j]), (j * 10 + 0) + (j * 10 + 1)) << "row " << j;
    }
}

// (c0 + c1) * t0 -- two operations, so the first must route through a temporary
// and the second must read it back.
TEST(FR_EVALUATOR, chainsThroughATemporary) {
    auto l = layout();
    auto constPols = grid(DOMAIN, N_CONST, 100);
    auto trace = grid(DOMAIN, N_TRACE, 0);

    StepsParams params;
    params.constPols = constPols.data();
    params.trace = trace.data();

    const uint16_t TMP = (uint16_t)l.tmpType();

    std::vector<uint16_t> args;
    // op 0: tmp[1] = const[0] + const[1]
    args.push_back((uint16_t)FrOp::add);
    args.push_back(1);
    operand(args, 0, 0, 0);
    operand(args, 0, 1, 0);
    // op 1: out = tmp[1] * trace[0]
    args.push_back((uint16_t)FrOp::mul);
    args.push_back(0);
    operand(args, TMP, 1, 0);
    operand(args, 1, 0, 0);

    std::vector<uint8_t> ops = {0, 0};
    ParserParams pp{2, 0, 2 * ARGS_PER_OP, 0};
    ParserArgs pa{ops.data(), args.data()};

    const uint64_t nrows = 4;
    std::vector<FrEl> tmp(4 * nrows), out(nrows);

    PilFflonk::evaluateExpression(l, params, pp, pa, tmp.data(), out.data(), 0, nrows, DOMAIN, false, false);

    for (uint64_t j = 0; j < nrows; j++) {
        const uint64_t c0 = 100 + j * 10 + 0;
        const uint64_t c1 = 100 + j * 10 + 1;
        const uint64_t t0 = j * 10 + 0;
        EXPECT_EQ(asUI(out[j]), (c0 + c1) * t0) << "row " << j;
    }
}

// The x' access: an operand with stride index 1 reads the next row.
TEST(FR_EVALUATOR, readsTheNextRowThroughStride) {
    auto l = layout();
    auto trace = grid(DOMAIN, N_TRACE, 0);

    StepsParams params;
    params.trace = trace.data();

    std::vector<uint16_t> args;
    args.push_back((uint16_t)FrOp::sub);
    args.push_back(0);
    operand(args, 1, 0, 1);   // trace column 0, next row
    operand(args, 1, 0, 0);   // trace column 0, current row
    std::vector<uint8_t> ops = {0};

    ParserParams pp{1, 0, ARGS_PER_OP, 0};
    ParserArgs pa{ops.data(), args.data()};

    const uint64_t nrows = 3;
    std::vector<FrEl> tmp(4 * nrows), out(nrows);

    PilFflonk::evaluateExpression(l, params, pp, pa, tmp.data(), out.data(), 0, nrows, DOMAIN, false, false);

    // Consecutive rows differ by 10 in this grid.
    for (uint64_t j = 0; j < nrows; j++) EXPECT_EQ(asUI(out[j]), 10u) << "row " << j;
}

// A constant operand is one whose type tag lies past the temporaries; it is a
// single value broadcast across the block.
TEST(FR_EVALUATOR, broadcastsANumberOperand) {
    auto l = layout();
    auto trace = grid(DOMAIN, N_TRACE, 0);
    std::vector<FrEl> numbers = {E.fr.set(1000)};

    StepsParams params;
    params.trace = trace.data();
    params.numbers = numbers.data();

    const uint16_t NUMBER = (uint16_t)(l.bufferCommitsSize() + 2);

    std::vector<uint16_t> args;
    args.push_back((uint16_t)FrOp::add);
    args.push_back(0);
    operand(args, 1, 1, 0);        // trace column 1
    operand(args, NUMBER, 0, 0);   // literal
    std::vector<uint8_t> ops = {0};

    ParserParams pp{1, 0, ARGS_PER_OP, 0};
    ParserArgs pa{ops.data(), args.data()};

    const uint64_t nrows = 4;
    std::vector<FrEl> tmp(4 * nrows), out(nrows);

    PilFflonk::evaluateExpression(l, params, pp, pa, tmp.data(), out.data(), 0, nrows, DOMAIN, false, false);

    for (uint64_t j = 0; j < nrows; j++) {
        EXPECT_EQ(asUI(out[j]), (j * 10 + 1) + 1000) << "row " << j;
    }
}

// Bytecode built for Goldilocks names dim-3 operand combinations. Running it
// over BN254 would read past the operands and silently corrupt results, so a
// non-zero dimension case is refused.
TEST(FR_EVALUATOR, refusesExtensionFieldBytecode) {
    auto l = layout();
    auto trace = grid(DOMAIN, N_TRACE, 0);

    StepsParams params;
    params.trace = trace.data();

    std::vector<uint16_t> args;
    args.push_back((uint16_t)FrOp::add);
    args.push_back(0);
    operand(args, 1, 0, 0);
    operand(args, 1, 1, 0);
    std::vector<uint8_t> ops = {1};  // dim3 (op) dim1

    ParserParams pp{1, 0, ARGS_PER_OP, 0};
    ParserArgs pa{ops.data(), args.data()};

    std::vector<FrEl> tmp(16), out(4);

    try {
        PilFflonk::evaluateExpression(l, params, pp, pa, tmp.data(), out.data(), 0, 4, DOMAIN, false, false);
        FAIL() << "expected a throw for dimension case 1";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("dim 1 throughout"), std::string::npos) << e.what();
    }
}

// A declared argument count that disagrees with what was consumed means the
// bytecode and the interpreter disagree about operation width.
TEST(FR_EVALUATOR, detectsAnArgumentCountMismatch) {
    auto l = layout();
    auto trace = grid(DOMAIN, N_TRACE, 0);

    StepsParams params;
    params.trace = trace.data();

    std::vector<uint16_t> args;
    args.push_back((uint16_t)FrOp::add);
    args.push_back(0);
    operand(args, 1, 0, 0);
    operand(args, 1, 1, 0);
    std::vector<uint8_t> ops = {0};

    ParserParams pp{1, 0, /*nArgs=*/99, 0};  // wrong on purpose
    ParserArgs pa{ops.data(), args.data()};

    std::vector<FrEl> tmp(16), out(4);
    EXPECT_ANY_THROW(
        PilFflonk::evaluateExpression(l, params, pp, pa, tmp.data(), out.data(), 0, 4, DOMAIN, false, false));
}

TEST(FR_EVALUATOR, rejectsUnboundBytecode) {
    auto l = layout();
    StepsParams params;
    ParserParams pp{1, 0, ARGS_PER_OP, 0};
    ParserArgs pa{nullptr, nullptr};
    std::vector<FrEl> tmp(16), out(4);

    EXPECT_ANY_THROW(
        PilFflonk::evaluateExpression(l, params, pp, pa, tmp.data(), out.data(), 0, 4, DOMAIN, false, false));
}
