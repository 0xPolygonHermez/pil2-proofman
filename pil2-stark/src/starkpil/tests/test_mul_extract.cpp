// Bytecode -> linear form tests: the extracted closed form must agree with an interpretation of the
// same bytecode on random column data, for every row -- that equality is the whole licence for the
// scatter kernel to skip the interpreter. Anything non-linear MUST come back `ok = false`, or the
// extractor would silently count the wrong rows.
//
// Build and run:
//   g++ -O2 -std=c++17 -DMUL_EXTRACT_TEST_MAIN -I src/starkpil \
//       -o /tmp/test_mul_extract src/starkpil/tests/test_mul_extract.cpp && /tmp/test_mul_extract
#include "../multiplicity_bytecode.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

#ifdef MUL_EXTRACT_TEST_MAIN

static const uint32_t BASE = 8;        // bufferCommitSize
static const uint32_t NUNIF = 4;       // entries per uniform pool
static const uint32_t SECTIONS = 3;    // const + cm1..cm3
static const uint64_t NROWS = 64;
static const uint32_t NCOLS = 4;

// Column data, indexed [type][col][row]. Type 0 is const pols, 1..SECTIONS the committed stages.
static uint64_t g_cols[SECTIONS + 1][NCOLS][NROWS];
// The uniform pools, indexed by operand type offset from BASE: +2 publics, +4 airvalues,
// +5 proofvalues, +6 airgroupvalues. Same value at every row -- that is what "uniform" means.
static uint64_t g_unif[9][NUNIF];

static uint64_t colAt(uint16_t type, uint16_t col, uint64_t row) { return g_cols[type][col][row]; }

// Reference: run the bytecode the way computeExpressions_ does, one row at a time.
static bool interpret(const MulByteCode& bc, uint64_t row, uint64_t& out) {
    std::vector<uint64_t> tmp(bc.nTemp1 + 1, 0);
    auto load = [&](uint16_t type, uint16_t idx, uint64_t& v) -> bool {
        if (type == bc.base) { v = tmp[idx]; return true; }
        if (type == bc.base + 3) { v = bc.numbers[idx]; return true; }
        if (mulIsUniformType(type, bc.base)) { v = g_unif[type - bc.base][idx]; return true; }
        if (type >= bc.base + 2 || type > bc.nSections) return false;
        v = colAt(type, idx, row);
        return true;
    };
    uint64_t i = 0, last = 0;
    for (uint32_t k = 0; k < bc.nOps; ++k) {
        uint64_t a = 0, b = 0, r = 0;
        if (!load(bc.args[i + 2], bc.args[i + 3], a)) return false;
        if (!load(bc.args[i + 5], bc.args[i + 6], b)) return false;
        switch (bc.args[i]) {
            case 0: r = mulAddFE(a, b); break;
            case 1: r = mulAddFE(a, b == 0 ? 0 : MUL_P - b); break;
            case 2: r = mulMulFE(a, b); break;
            case 3: r = mulAddFE(b, a == 0 ? 0 : MUL_P - a); break;
            default: return false;
        }
        if (k + 1 == bc.nOps) last = r; else tmp[bc.args[i + 1]] = r;
        i += 8;
    }
    out = last;
    return true;
}

// Evaluate an extracted form the way the kernel does.
static uint64_t evalForm(const MulLinForm& f, uint64_t row) {
    uint64_t acc = f.konst;
    for (uint8_t i = 0; i < f.n; ++i) {
        const uint64_t v = mulIsUniformType(f.t[i].type, BASE)
                         ? g_unif[f.t[i].type - BASE][f.t[i].argIdx]
                         : colAt(f.t[i].type, f.t[i].argIdx, row);
        acc = mulAddFE(acc, f.t[i].coef == 1 ? v : mulMulFE(v, f.t[i].coef));
    }
    return acc;
}

static bool agreesEverywhere(const MulByteCode& bc, const MulLinForm& f) {
    if (!f.ok) return false;
    for (uint64_t r = 0; r < NROWS; ++r) {
        uint64_t ref = 0;
        if (!interpret(bc, r, ref)) return false;
        if (evalForm(f, r) != ref) return false;
    }
    return true;
}

// arg tuples are (op, dst, type0, idx0, off0, type1, idx1, off1)
static MulByteCode makeBC(const std::vector<uint8_t>& ops, const std::vector<uint16_t>& args,
                          const std::vector<uint64_t>& numbers, uint32_t nTemp1) {
    MulByteCode bc;
    bc.ops = ops.data(); bc.args = args.data();
    bc.numbers = numbers.empty() ? nullptr : numbers.data();
    bc.nOps = (uint32_t)ops.size(); bc.nTemp1 = nTemp1;
    bc.base = BASE; bc.nSections = SECTIONS;
    return bc;
}

int main() {
    srand(1);
    for (uint32_t t = 0; t <= SECTIONS; ++t)
        for (uint32_t c = 0; c < NCOLS; ++c)
            for (uint64_t r = 0; r < NROWS; ++r)
                g_cols[t][c][r] = ((uint64_t)rand() << 20) ^ (uint64_t)rand();

    for (uint32_t t = 0; t < 9; ++t)
        for (uint32_t i = 0; i < NUNIF; ++i) g_unif[t][i] = ((uint64_t)rand() << 20) ^ (uint64_t)rand();

    // ---- cm1[0] - 7: the shape `expression - min` compiles to -----------------------------
    {
        std::vector<uint64_t> nums = {7};
        std::vector<uint8_t>  ops  = {0};
        std::vector<uint16_t> args = {1, 0, 1, 0, 0, BASE + 3, 0, 0};
        MulByteCode bc = makeBC(ops, args, nums, 0);
        MulLinForm f = mulWalkBytecode(bc);
        check(agreesEverywhere(bc, f), "cm1[0] - 7 matches the interpreter");
        check(f.n == 1 && f.konst == MUL_P - 7, "cm1[0] - 7 reduced to one term");
    }

    // ---- 2^16 - cm1[1]: the `max - expression` direction (rsub) ----------------------------
    {
        std::vector<uint64_t> nums = {65536};
        std::vector<uint8_t>  ops  = {0};
        std::vector<uint16_t> args = {3, 0, 1, 1, 0, BASE + 3, 0, 0};
        MulByteCode bc = makeBC(ops, args, nums, 0);
        check(agreesEverywhere(bc, mulWalkBytecode(bc)), "65536 - cm1[1] matches the interpreter");
    }

    // ---- a - b - 1 across two ops, through a temporary -------------------------------------
    {
        std::vector<uint64_t> nums = {1};
        std::vector<uint8_t>  ops  = {0, 0};
        std::vector<uint16_t> args = {1, 0, 1, 0, 0, 1, 1, 0,            // tmp0 = cm1[0] - cm1[1]
                                      1, 0, BASE, 0, 0, BASE + 3, 0, 0}; // tmp0 - 1
        MulByteCode bc = makeBC(ops, args, nums, 1);
        MulLinForm f = mulWalkBytecode(bc);
        check(agreesEverywhere(bc, f), "cm1[0] - cm1[1] - 1 matches the interpreter");
        check(f.n == 2, "cm1[0] - cm1[1] - 1 kept both columns");
    }

    // ---- 256*cm2[0] + cm2[1]: a scaled term, the affine shape ------------------------------
    {
        std::vector<uint64_t> nums = {256};
        std::vector<uint8_t>  ops  = {0, 0};
        std::vector<uint16_t> args = {2, 0, BASE + 3, 0, 0, 2, 0, 0,     // tmp0 = 256 * cm2[0]
                                      0, 0, BASE, 0, 0, 2, 1, 0};        // tmp0 + cm2[1]
        MulByteCode bc = makeBC(ops, args, nums, 1);
        MulLinForm f = mulWalkBytecode(bc);
        check(agreesEverywhere(bc, f), "256*cm2[0] + cm2[1] matches the interpreter");
        check(f.n == 2 && f.t[0].coef == 256, "256*cm2[0] + cm2[1] kept the coefficient");
    }

    // ---- the same column twice: folded to one load ----------------------------------------
    {
        std::vector<uint8_t>  ops  = {0};
        std::vector<uint16_t> args = {0, 0, 1, 2, 0, 1, 2, 0};
        MulByteCode bc = makeBC(ops, args, {}, 0);
        MulLinForm f = mulWalkBytecode(bc);
        check(agreesEverywhere(bc, f), "cm1[2] + cm1[2] matches the interpreter");
        check(f.n == 1 && f.t[0].coef == 2, "cm1[2] + cm1[2] folded into one term");
    }

    // ---- a public as an operand: a runtime constant, so a term rather than a give-up ------
    {
        std::vector<uint8_t>  ops  = {0};
        std::vector<uint16_t> args = {1, 0, 1, 0, 0, BASE + 2, 1, 0};   // cm1[0] - public[1]
        MulByteCode bc = makeBC(ops, args, {}, 0);
        MulLinForm f = mulWalkBytecode(bc);
        check(agreesEverywhere(bc, f), "cm1[0] - public[1] matches the interpreter");
        check(f.n == 2, "cm1[0] - public[1] kept the public as a term");
    }

    // ---- an airvalue, and a uniform operand on both sides ---------------------------------
    {
        std::vector<uint8_t>  ops  = {0};
        std::vector<uint16_t> args = {0, 0, BASE + 4, 2, 0, BASE + 6, 3, 0};
        MulByteCode bc = makeBC(ops, args, {}, 0);
        check(agreesEverywhere(bc, mulWalkBytecode(bc)), "airvalue[2] + airgroupvalue[3] matches");
    }

    // ---- a uniform pool the scatter cannot read yet: still refused ------------------------
    {
        std::vector<uint8_t>  ops  = {0};
        std::vector<uint16_t> ch   = {0, 0, 1, 0, 0, BASE + 7, 0, 0};   // a challenge
        check(!mulWalkBytecode(makeBC(ops, ch, {}, 0)).ok, "challenge operand refused");
        std::vector<uint16_t> ev   = {0, 0, 1, 0, 0, BASE + 8, 0, 0};   // an eval
        check(!mulWalkBytecode(makeBC(ops, ev, {}, 0)).ok, "eval operand refused");
    }

    // ---- a column times a column: NOT linear, must be refused -----------------------------
    {
        std::vector<uint8_t>  ops  = {0};
        std::vector<uint16_t> args = {2, 0, 1, 0, 0, 1, 1, 0};
        MulByteCode bc = makeBC(ops, args, {}, 0);
        check(!mulWalkBytecode(bc).ok, "cm1[0] * cm1[1] refused");
    }

    // ---- an operand the walk does not model: must be refused ------------------------------
    {
        std::vector<uint8_t>  ops  = {0};
        std::vector<uint16_t> zi = {0, 0, 1, 0, 0, SECTIONS + 1, 0, 0}; // zi, past the last section
        check(!mulWalkBytecode(makeBC(ops, zi, {}, 0)).ok, "zi operand refused");

        std::vector<uint8_t> dim3 = {1};                                 // dim3 op, not a range check
        std::vector<uint16_t> a3 = {0, 0, 1, 0, 0, 1, 1, 0};
        check(!mulWalkBytecode(makeBC(dim3, a3, {}, 0)).ok, "dim3 operation refused");
    }

    // ---- more distinct columns than the form can hold: refused, not truncated --------------
    {
        std::vector<uint8_t>  ops(MUL_MAX_TERMS, 0);
        std::vector<uint16_t> args;
        args.insert(args.end(), {0, 0, 1, 0, 0, 1, 1, 0});               // tmp0 = c0 + c1
        for (uint32_t k = 1; k < MUL_MAX_TERMS; ++k) {                   // keep adding new columns
            const uint16_t type = (uint16_t)(1 + (k % SECTIONS));
            args.insert(args.end(), {0, 0, BASE, 0, 0, type, (uint16_t)(k % NCOLS), 0});
        }
        check(!mulWalkBytecode(makeBC(ops, args, {}, 1)).ok, "overlong term list refused");
    }

    printf(failures == 0 ? "\ntest_mul_extract: all passed\n" : "\ntest_mul_extract: %d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
#endif
