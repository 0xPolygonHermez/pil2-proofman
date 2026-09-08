// Domain-driver tests: evaluate an expression over every row of a trace.
//
// These are the first tests that exercise the evaluator the way a prover would
// -- a real constraint, over a whole domain, with the wrap at the boundary that
// makes a next-row access on the final row read row zero.
#include <gtest/gtest.h>
#include <vector>

#include "fr_domain.hpp"

using PilFflonk::Dest;
using PilFflonk::ExpressionsLayout;
using PilFflonk::FrEl;
using PilFflonk::FrOp;
using PilFflonk::ParserArgs;
using PilFflonk::ParserParams;
using PilFflonk::StepsParams;

static AltBn128::Engine &E = AltBn128::Engine::engine;

namespace {

constexpr uint64_t N_TRACE = 2;

ExpressionsLayout layout() {
    ExpressionsLayout l;
    l.nStages = 1;
    l.nConstants = 1;
    l.mapSectionsN = {1, N_TRACE, 0};
    l.mapOffsets = {0, 0, 0};
    l.nextStrides = {0, 1};
    return l;
}

void operand(std::vector<uint16_t> &v, uint16_t type, uint16_t pos, uint16_t stride) {
    v.push_back(type);
    v.push_back(pos);
    v.push_back(stride);
}

uint64_t asUI(const FrEl &e) { return std::stoull(E.fr.toString(e)); }

// trace[r][c] = r * 10 + c
std::vector<FrEl> trace(uint64_t rows) {
    std::vector<FrEl> t(rows * N_TRACE);
    for (uint64_t r = 0; r < rows; r++)
        for (uint64_t c = 0; c < N_TRACE; c++) t[r * N_TRACE + c] = E.fr.set(r * 10 + c);
    return t;
}

} // namespace

// out[r] = trace[r][0] + trace[r][1], across the whole domain.
TEST(FR_DOMAIN, evaluatesEveryRow) {
    const uint64_t domain = 64;
    auto l = layout();
    auto t = trace(domain);

    StepsParams params;
    params.trace = t.data();

    std::vector<uint16_t> args;
    args.push_back((uint16_t)FrOp::add);
    args.push_back(0);
    operand(args, 1, 0, 0);
    operand(args, 1, 1, 0);
    std::vector<uint8_t> ops = {0};

    ParserParams pp{1, 0, PilFflonk::ARGS_PER_OP, 0};
    ParserArgs pa{ops.data(), args.data()};

    std::vector<FrEl> result(domain, E.fr.zero());
    Dest dest{result.data(), 0, domain};

    PilFflonk::calculateExpression(l, params, pp, pa, dest, domain, /*maxTmp=*/4, false);

    for (uint64_t r = 0; r < domain; r++) {
        EXPECT_EQ(asUI(result[r]), (r * 10) + (r * 10 + 1)) << "row " << r;
    }
}

// The defining edge case: a next-row access on the final row must read row 0.
// Getting this wrong reads past the trace, which is both a correctness bug and
// an out-of-bounds read.
TEST(FR_DOMAIN, wrapsTheNextRowAccessAtTheBoundary) {
    const uint64_t domain = 16;
    auto l = layout();
    auto t = trace(domain);

    StepsParams params;
    params.trace = t.data();

    // out[r] = trace[r+1][0]
    std::vector<uint16_t> args;
    args.push_back((uint16_t)FrOp::add);
    args.push_back(0);
    operand(args, 1, 0, 1);   // next row, column 0
    operand(args, 1, 0, 1);   // same, so the result is 2 * trace[r+1][0]
    std::vector<uint8_t> ops = {0};

    ParserParams pp{1, 0, PilFflonk::ARGS_PER_OP, 0};
    ParserArgs pa{ops.data(), args.data()};

    std::vector<FrEl> result(domain, E.fr.zero());
    Dest dest{result.data(), 0, domain};

    PilFflonk::calculateExpression(l, params, pp, pa, dest, domain, 4, false, /*nrowsPack=*/4);

    for (uint64_t r = 0; r < domain - 1; r++) {
        EXPECT_EQ(asUI(result[r]), 2 * ((r + 1) * 10)) << "row " << r;
    }
    EXPECT_EQ(asUI(result[domain - 1]), 0u) << "the final row must wrap to row 0, whose column 0 is 0";
}

// The block size must not change the answer -- it is a performance knob only.
TEST(FR_DOMAIN, resultIsIndependentOfBlockSize) {
    const uint64_t domain = 32;
    auto l = layout();
    auto t = trace(domain);

    StepsParams params;
    params.trace = t.data();

    std::vector<uint16_t> args;
    args.push_back((uint16_t)FrOp::mul);
    args.push_back(0);
    operand(args, 1, 0, 1);   // next row
    operand(args, 1, 1, 0);   // current row
    std::vector<uint8_t> ops = {0};

    ParserParams pp{1, 0, PilFflonk::ARGS_PER_OP, 0};
    ParserArgs pa{ops.data(), args.data()};

    std::vector<std::vector<FrEl>> results;
    for (uint64_t pack : {1u, 3u, 8u, 32u, 128u}) {
        std::vector<FrEl> r(domain, E.fr.zero());
        Dest dest{r.data(), 0, domain};
        PilFflonk::calculateExpression(l, params, pp, pa, dest, domain, 4, false, pack);
        results.push_back(std::move(r));
    }

    for (size_t i = 1; i < results.size(); i++) {
        for (uint64_t r = 0; r < domain; r++) {
            ASSERT_TRUE(E.fr.eq(results[0][r], results[i][r]))
                << "block size changed the result at row " << r << " (variant " << i << ")";
        }
    }
}

// A domain that is not a multiple of the block size must still be covered
// exactly, with no row skipped and none written twice.
TEST(FR_DOMAIN, handlesADomainThatIsNotAMultipleOfTheBlock) {
    const uint64_t domain = 10;  // not a multiple of 4
    auto l = layout();
    auto t = trace(domain);

    StepsParams params;
    params.trace = t.data();

    std::vector<uint16_t> args;
    args.push_back((uint16_t)FrOp::add);
    args.push_back(0);
    operand(args, 1, 1, 0);
    operand(args, 1, 1, 0);
    std::vector<uint8_t> ops = {0};

    ParserParams pp{1, 0, PilFflonk::ARGS_PER_OP, 0};
    ParserArgs pa{ops.data(), args.data()};

    std::vector<FrEl> result(domain, E.fr.zero());
    Dest dest{result.data(), 0, domain};

    PilFflonk::calculateExpression(l, params, pp, pa, dest, domain, 4, false, /*nrowsPack=*/4);

    for (uint64_t r = 0; r < domain; r++) {
        EXPECT_EQ(asUI(result[r]), 2 * (r * 10 + 1)) << "row " << r << " not covered correctly";
    }
}

// Writing into one column of a wider destination must leave its neighbours
// untouched, which is how an intermediate polynomial joins a trace.
TEST(FR_DOMAIN, writesIntoOneColumnOfAWiderDestination) {
    const uint64_t domain = 8;
    const uint64_t destCols = 3;
    auto l = layout();
    auto t = trace(domain);

    StepsParams params;
    params.trace = t.data();

    std::vector<uint16_t> args;
    args.push_back((uint16_t)FrOp::add);
    args.push_back(0);
    operand(args, 1, 0, 0);
    operand(args, 1, 1, 0);
    std::vector<uint8_t> ops = {0};

    ParserParams pp{1, 0, PilFflonk::ARGS_PER_OP, 0};
    ParserArgs pa{ops.data(), args.data()};

    std::vector<FrEl> buffer(domain * destCols, E.fr.zero());
    Dest dest{buffer.data(), destCols, domain};

    PilFflonk::calculateExpression(l, params, pp, pa, dest, domain, 4, false);

    for (uint64_t r = 0; r < domain; r++) {
        EXPECT_EQ(asUI(buffer[r * destCols]), (r * 10) + (r * 10 + 1)) << "row " << r;
        EXPECT_EQ(asUI(buffer[r * destCols + 1]), 0u) << "neighbouring column touched at row " << r;
        EXPECT_EQ(asUI(buffer[r * destCols + 2]), 0u) << "neighbouring column touched at row " << r;
    }
}

TEST(FR_DOMAIN, emptyDomainIsANoOp) {
    auto l = layout();
    StepsParams params;
    std::vector<uint8_t> ops = {0};
    std::vector<uint16_t> args(PilFflonk::ARGS_PER_OP, 0);
    ParserParams pp{1, 0, PilFflonk::ARGS_PER_OP, 0};
    ParserArgs pa{ops.data(), args.data()};
    Dest dest;

    EXPECT_NO_THROW(PilFflonk::calculateExpression(l, params, pp, pa, dest, 0, 4, false));
}
