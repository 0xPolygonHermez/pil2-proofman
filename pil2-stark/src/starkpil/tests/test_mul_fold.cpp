// Fold-shape tests: `konst + sum(coef[e] * element_e)` must emit a program that evaluates to
// exactly that; a wrong temporary silently counts a wrong table row. SetupCtx cannot be built
// here, so this drives the emit helpers on the shapes `mulCompileFold` produces.
//
// Build and run:
//   g++ -O2 -std=c++17 -DMUL_FOLD_TEST_MAIN -I src/starkpil \
//       -o /tmp/test_mul_fold src/starkpil/tests/test_mul_fold.cpp && /tmp/test_mul_fold
#include "../multiplicity_job.hpp"
#include <cstdio>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

#ifdef MUL_FOLD_TEST_MAIN

static uint64_t g_col[8] = { 3, 5, 7, 11, 13, 17, 19, 23 };

// The evaluator both backends run, reduced to what this needs (mulEvalProgram /
// mulEvalProgramCPU): SSA temporaries, and the LAST instruction's result is the value, unwritten.
static uint64_t evalProgram(const MulProgram& p) {
    uint64_t tmp[MUL_PROG_MAX_TEMP] = {0};
    auto val = [&](const MulOperandDev& o) -> uint64_t {
        if (o.kind == MUL_OPND_TEMP)  return tmp[o.tmp];
        if (o.kind == MUL_OPND_CONST) return o.konst;
        return g_col[o.term.col];
    };
    uint64_t last = 0;
    for (size_t k = 0; k < p.insns.size(); ++k) {
        const MulInsnDev& in = p.insns[k];
        const uint64_t a = val(in.a), b = val(in.b);
        uint64_t r;
        switch (in.op) {
            case 0:  r = mulAddFE(a, b); break;
            case 1:  r = mulSubFEHD(a, b); break;
            case 2:  r = mulMulFE(a, b); break;
            default: r = mulSubFEHD(b, a); break;
        }
        if (k + 1 == p.insns.size()) { last = r; break; }
        if (in.dst >= MUL_PROG_MAX_TEMP) { printf("  temp %u out of range\n", in.dst); ++failures; return 0; }
        tmp[in.dst] = r;
    }
    return last;
}

static MulOperandDev opCol(uint16_t c) {
    MulOperandDev o{}; o.kind = MUL_OPND_COL; o.term.col = c; return o;
}

// What mulCompileFold emits for operand elements, once they are resolved (a spliced expression
// element is not modelled here). Same temporaries as the real one: the top of the temp space.
static bool buildFold(const uint16_t* col, const uint64_t* coef, uint8_t n, uint64_t konst,
                      MulProgram& out) {
    out = MulProgram{};
    const uint16_t SUM = MUL_PROG_MAX_TEMP - 1, SCR = MUL_PROG_MAX_TEMP - 2;
    bool haveSum = false;
    for (uint8_t e = 0; e < n; ++e) {
        if (coef[e] == 0) continue;
        MulOperandDev o = opCol(col[e]);
        if (coef[e] != 1) { mulEmit(out, SCR, o, 2, mulOpConst(coef[e])); o = mulOpTemp(SCR); }
        if (!haveSum) { mulEmit(out, SUM, o, 0, mulOpConst(0)); haveSum = true; }
        else          { mulEmit(out, SUM, mulOpTemp(SUM), 0, o); }
    }
    if (!haveSum) return false;
    if (konst != 0) mulEmit(out, SUM, mulOpTemp(SUM), 0, mulOpConst(konst));
    out.ok = true;
    return true;
}

int main() {
    printf("test_mul_fold\n");
    {   // the shape the zisk PIL actually fits: two columns, one scaled
        const uint16_t col[2] = {0, 1}; const uint64_t coef[2] = {256, 1};
        MulProgram out;
        check(buildFold(col, coef, 2, 0, out), "two-column fold builds");
        check(evalProgram(out) == 3 * 256 + 5, "two-column fold evaluates");
        bool inRange = true;
        for (const auto& in : out.insns) {
            if (in.dst >= MUL_PROG_MAX_TEMP) inRange = false;
            if (in.a.kind == MUL_OPND_TEMP && in.a.tmp >= MUL_PROG_MAX_TEMP) inRange = false;
            if (in.b.kind == MUL_OPND_TEMP && in.b.tmp >= MUL_PROG_MAX_TEMP) inRange = false;
        }
        check(inRange, "every temporary index stays inside the cap");
    }
    {   // the scratch temporary must not survive into the running sum
        const uint16_t col[3] = {0, 1, 2}; const uint64_t coef[3] = {2, 4, 8};
        MulProgram out;
        check(buildFold(col, coef, 3, 0, out), "all-scaled fold builds");
        check(evalProgram(out) == 3 * 2 + 5 * 4 + 7 * 8, "scaling does not clobber the sum");
    }
    {   // a coefficient of zero drops the element: the fit found that column irrelevant
        const uint16_t col[3] = {0, 1, 2}; const uint64_t coef[3] = {1, 0, 2};
        MulProgram out;
        check(buildFold(col, coef, 3, 0, out), "zero-coefficient fold builds");
        check(evalProgram(out) == 3 + 7 * 2, "zero-coefficient element is skipped");
    }
    {   // konst is added once, at the end
        const uint16_t col[2] = {0, 1}; const uint64_t coef[2] = {1, 1};
        MulProgram out;
        check(buildFold(col, coef, 2, 9, out), "fold with a constant builds");
        check(evalProgram(out) == 3 + 5 + 9, "constant is added once");
    }
    {   // coefficients are field elements, not small integers: a negative one arrives near p
        const uint16_t col[2] = {0, 1}; const uint64_t coef[2] = {1, MUL_P - 1};
        MulProgram out;
        check(buildFold(col, coef, 2, 0, out), "field-element coefficient folds");
        check(evalProgram(out) == mulAddFE(3, MUL_P - 5), "negative coefficient wraps through p");
    }
    {   // nothing to fold is a refusal, not an empty program the evaluator would read past
        const uint16_t col[2] = {0, 1}; const uint64_t coef[2] = {0, 0};
        MulProgram out;
        check(!buildFold(col, coef, 2, 5, out), "an all-zero fold is refused");
    }
    printf("\ntest_mul_fold: %s\n", failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}

#endif
