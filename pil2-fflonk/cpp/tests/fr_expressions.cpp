// Operand addressing tests for the BN254 expression evaluator.
//
// Addressing bugs here are the dangerous kind: reading the wrong column or the
// wrong row still yields a valid field element, so a constraint evaluates to
// something plausible and wrong. Each case pins an index computation.
#include <gtest/gtest.h>
#include <vector>

#include "fr_expressions.hpp"

using PilFflonk::ExpressionsLayout;
using PilFflonk::FrEl;
using PilFflonk::StepsParams;

static AltBn128::Engine &E = AltBn128::Engine::engine;

namespace {

constexpr uint64_t DOMAIN = 8;
constexpr uint64_t N_CONSTANTS = 3;
constexpr uint64_t N_STAGE1 = 2;

ExpressionsLayout layout() {
    ExpressionsLayout l;
    l.nStages = 2;
    l.nConstants = N_CONSTANTS;
    // Type tags: 0 = const, 1..nStages+1 = committed stages.
    l.mapSectionsN = {N_CONSTANTS, N_STAGE1, 4, 0};
    l.mapOffsets = {0, 0, 100, 0};
    l.nextStrides = {0, 1};  // current row, next row
    return l;
}

// value(row, col) = row * 100 + col, so a misread is obvious.
std::vector<FrEl> grid(uint64_t rows, uint64_t cols, uint64_t bias = 0) {
    std::vector<FrEl> v(rows * cols);
    for (uint64_t r = 0; r < rows; r++)
        for (uint64_t c = 0; c < cols; c++) v[r * cols + c] = E.fr.set(bias + r * 100 + c);
    return v;
}

uint64_t asUI(const FrEl &e) { return std::stoull(E.fr.toString(e)); }

} // namespace

TEST(FR_EXPRESSIONS, readsConstantPolynomialColumn) {
    auto l = layout();
    auto constPols = grid(DOMAIN, N_CONSTANTS);

    StepsParams params;
    params.constPols = constPols.data();

    // type 0, column 2, stride index 0 (offset 0)
    const uint16_t args[] = {0, 2, 0};
    std::vector<FrEl> value(4);

    const FrEl *got = PilFflonk::load(l, params, value.data(), nullptr, args, 0, /*row=*/1, /*nrowsPack=*/4, DOMAIN,
                                      false, false);

    for (uint64_t j = 0; j < 4; j++) {
        EXPECT_EQ(asUI(got[j]), (1 + j) * 100 + 2) << "row " << j;
    }
}

TEST(FR_EXPRESSIONS, appliesNextRowStride) {
    auto l = layout();
    auto constPols = grid(DOMAIN, N_CONSTANTS);

    StepsParams params;
    params.constPols = constPols.data();

    // stride index 1 -> offset 1, i.e. the x' access
    const uint16_t args[] = {0, 1, 1};
    std::vector<FrEl> value(3);

    const FrEl *got = PilFflonk::load(l, params, value.data(), nullptr, args, 0, /*row=*/2, /*nrowsPack=*/3, DOMAIN,
                                      false, false);

    for (uint64_t j = 0; j < 3; j++) {
        EXPECT_EQ(asUI(got[j]), (2 + j + 1) * 100 + 1) << "row " << j;
    }
}

// A cyclic read wraps at the domain boundary; a non-cyclic one runs off the end.
// This is what makes the last rows of a trace read the first rows for x'.
TEST(FR_EXPRESSIONS, cyclicReadWrapsAtTheDomainBoundary) {
    auto l = layout();
    auto constPols = grid(DOMAIN, N_CONSTANTS);

    StepsParams params;
    params.constPols = constPols.data();

    const uint16_t args[] = {0, 0, 1};  // column 0, next row
    std::vector<FrEl> value(2);

    // Rows 6 and 7 with a +1 offset: 7 stays in range, 8 must wrap to 0.
    const FrEl *got = PilFflonk::load(l, params, value.data(), nullptr, args, 0, /*row=*/6, /*nrowsPack=*/2, DOMAIN,
                                      false, /*isCyclic=*/true);

    EXPECT_EQ(asUI(got[0]), 7 * 100 + 0);
    EXPECT_EQ(asUI(got[1]), 0 * 100 + 0) << "row 8 must wrap to row 0";
}

TEST(FR_EXPRESSIONS, readsStageOneFromTrace) {
    auto l = layout();
    auto trace = grid(DOMAIN, N_STAGE1, 7000);

    StepsParams params;
    params.trace = trace.data();

    const uint16_t args[] = {1, 1, 0};  // stage 1, column 1
    std::vector<FrEl> value(4);

    const FrEl *got =
        PilFflonk::load(l, params, value.data(), nullptr, args, 0, /*row=*/0, /*nrowsPack=*/4, DOMAIN, false, false);

    for (uint64_t j = 0; j < 4; j++) {
        EXPECT_EQ(asUI(got[j]), 7000 + j * 100 + 1) << "row " << j;
    }
}

// Later stages live inside aux_trace at a mapped offset, not at index 0.
TEST(FR_EXPRESSIONS, readsLaterStageAtItsMappedOffset) {
    auto l = layout();
    std::vector<FrEl> aux(100 + DOMAIN * 4);
    for (uint64_t r = 0; r < DOMAIN; r++)
        for (uint64_t c = 0; c < 4; c++) aux[100 + r * 4 + c] = E.fr.set(5000 + r * 100 + c);

    StepsParams params;
    params.aux_trace = aux.data();

    const uint16_t args[] = {2, 3, 0};  // stage 2, column 3
    std::vector<FrEl> value(2);

    const FrEl *got =
        PilFflonk::load(l, params, value.data(), nullptr, args, 0, /*row=*/5, /*nrowsPack=*/2, DOMAIN, false, false);

    EXPECT_EQ(asUI(got[0]), 5000 + 5 * 100 + 3);
    EXPECT_EQ(asUI(got[1]), 5000 + 6 * 100 + 3);
}

// The extended domain swaps which constant-polynomial buffer is read.
TEST(FR_EXPRESSIONS, extendedDomainSelectsTheExtendedConstants) {
    auto l = layout();
    auto base = grid(DOMAIN, N_CONSTANTS, 0);
    auto ext = grid(DOMAIN, N_CONSTANTS, 900000);

    StepsParams params;
    params.constPols = base.data();
    params.constPolsExt = ext.data();

    const uint16_t args[] = {0, 0, 0};
    std::vector<FrEl> value(1);

    const FrEl *plain =
        PilFflonk::load(l, params, value.data(), nullptr, args, 0, 3, 1, DOMAIN, /*domainExtended=*/false, false);
    EXPECT_EQ(asUI(plain[0]), 300);

    const FrEl *extended =
        PilFflonk::load(l, params, value.data(), nullptr, args, 0, 3, 1, DOMAIN, /*domainExtended=*/true, false);
    EXPECT_EQ(asUI(extended[0]), 900000 + 300);
}

// Temporaries are already contiguous: load returns a pointer into them rather
// than copying.
TEST(FR_EXPRESSIONS, returnsTemporariesInPlace) {
    auto l = layout();
    const uint64_t nrowsPack = 4;
    std::vector<FrEl> tmp(3 * nrowsPack);
    for (uint64_t i = 0; i < tmp.size(); i++) tmp[i] = E.fr.set(i + 1);
    FrEl *tmpBuffers[1] = {tmp.data()};

    StepsParams params;
    const uint16_t args[] = {(uint16_t)l.tmpType(), 2, 0};  // temporary index 2
    std::vector<FrEl> value(nrowsPack);

    const FrEl *got =
        PilFflonk::load(l, params, value.data(), tmpBuffers, args, 0, 0, nrowsPack, DOMAIN, false, false);

    EXPECT_EQ(got, &tmp[2 * nrowsPack]) << "temporaries must be returned in place, not copied";
    EXPECT_EQ(asUI(got[0]), 2 * nrowsPack + 1);
}

// FRI-specific operands cannot occur over BN254 and must fail loudly rather
// than read some other buffer.
TEST(FR_EXPRESSIONS, rejectsFriOnlyOperands) {
    auto l = layout();
    StepsParams params;
    std::vector<FrEl> value(4);

    const uint16_t xiArgs[] = {(uint16_t)l.xiType(), 0, 0};
    EXPECT_ANY_THROW(PilFflonk::load(l, params, value.data(), nullptr, xiArgs, 0, 0, 4, DOMAIN, false, false));

    const uint16_t tmp3Args[] = {(uint16_t)l.tmp3Type(), 0, 0};
    EXPECT_ANY_THROW(PilFflonk::load(l, params, value.data(), nullptr, tmp3Args, 0, 0, 4, DOMAIN, false, false));
}

TEST(FR_EXPRESSIONS, rejectsUnboundBuffers) {
    auto l = layout();
    StepsParams params;  // nothing bound
    std::vector<FrEl> value(4);

    const uint16_t constArgs[] = {0, 0, 0};
    EXPECT_ANY_THROW(PilFflonk::load(l, params, value.data(), nullptr, constArgs, 0, 0, 4, DOMAIN, false, false));

    const uint16_t traceArgs[] = {1, 0, 0};
    EXPECT_ANY_THROW(PilFflonk::load(l, params, value.data(), nullptr, traceArgs, 0, 0, 4, DOMAIN, false, false));
}
