// Tests for the BN254 field-operation primitive the expression evaluator sits
// on. Every operand combination the bytecode can emit is checked against direct
// engine calls, because a wrong opcode mapping here would corrupt every
// constraint evaluation while still producing well-formed field elements.
#include <gtest/gtest.h>
#include <vector>

#include "fr_pack.hpp"

using PilFflonk::FrEl;
using PilFflonk::FrOp;
using PilFflonk::NROWS_PACK;

static AltBn128::Engine &E = AltBn128::Engine::engine;

namespace {

std::vector<FrEl> seq(uint64_t n, uint64_t base) {
    std::vector<FrEl> v(n);
    for (uint64_t i = 0; i < n; i++) v[i] = E.fr.set(base + i * 3 + 1);
    return v;
}

} // namespace

TEST(FR_PACK, opcodesMatchDirectArithmetic) {
    const uint64_t n = NROWS_PACK;
    auto a = seq(n, 10);
    auto b = seq(n, 500);
    std::vector<FrEl> c(n);

    struct Case {
        uint64_t op;
        const char *name;
    };
    const Case cases[] = {
        {(uint64_t)FrOp::add, "add"},
        {(uint64_t)FrOp::sub, "sub"},
        {(uint64_t)FrOp::mul, "mul"},
        {(uint64_t)FrOp::sub_rev, "sub_rev"},
    };

    for (const auto &k : cases) {
        PilFflonk::op_pack(E, n, k.op, c.data(), a.data(), b.data());
        for (uint64_t i = 0; i < n; i++) {
            FrEl want;
            switch (k.op) {
                case 0: want = E.fr.add(a[i], b[i]); break;
                case 1: want = E.fr.sub(a[i], b[i]); break;
                case 2: want = E.fr.mul(a[i], b[i]); break;
                default: want = E.fr.sub(b[i], a[i]); break;
            }
            ASSERT_TRUE(E.fr.eq(c[i], want)) << k.name << " mismatch at row " << i;
        }
    }
}

// sub and sub_rev must not be interchangeable, or a swapped opcode would go
// unnoticed on symmetric inputs.
TEST(FR_PACK, subAndSubRevAreDistinct) {
    const uint64_t n = 8;
    auto a = seq(n, 1);
    auto b = seq(n, 99);
    std::vector<FrEl> fwd(n), rev(n);

    PilFflonk::op_pack(E, n, (uint64_t)FrOp::sub, fwd.data(), a.data(), b.data());
    PilFflonk::op_pack(E, n, (uint64_t)FrOp::sub_rev, rev.data(), a.data(), b.data());

    for (uint64_t i = 0; i < n; i++) {
        ASSERT_FALSE(E.fr.eq(fwd[i], rev[i])) << "sub and sub_rev agree at row " << i;
        ASSERT_TRUE(E.fr.eq(fwd[i], E.fr.neg(rev[i]))) << "sub_rev is not the negation of sub at row " << i;
    }
}

// A constant operand is broadcast across the block. This is the common path:
// challenges and constants are marked const by the evaluator.
TEST(FR_PACK, broadcastsConstantOperands) {
    const uint64_t n = 16;
    auto a = seq(n, 7);
    auto b = seq(n, 21);
    std::vector<FrEl> got(n);

    for (uint64_t op = 0; op <= 3; op++) {
        // const a: a[0] against every b[i].
        PilFflonk::op_pack(E, n, op, got.data(), a.data(), true, b.data(), false);
        for (uint64_t i = 0; i < n; i++) {
            FrEl want;
            PilFflonk::op_pack(E, 1, op, &want, &a[0], &b[i]);
            ASSERT_TRUE(E.fr.eq(got[i], want)) << "const-a op " << op << " at row " << i;
        }

        // const b: every a[i] against b[0].
        PilFflonk::op_pack(E, n, op, got.data(), a.data(), false, b.data(), true);
        for (uint64_t i = 0; i < n; i++) {
            FrEl want;
            PilFflonk::op_pack(E, 1, op, &want, &a[i], &b[0]);
            ASSERT_TRUE(E.fr.eq(got[i], want)) << "const-b op " << op << " at row " << i;
        }

        // both const: one value, repeated.
        PilFflonk::op_pack(E, n, op, got.data(), a.data(), true, b.data(), true);
        FrEl want;
        PilFflonk::op_pack(E, 1, op, &want, &a[0], &b[0]);
        for (uint64_t i = 0; i < n; i++) {
            ASSERT_TRUE(E.fr.eq(got[i], want)) << "const-both op " << op << " at row " << i;
        }
    }
}

// The non-const overload must agree with the const overload told nothing is
// constant -- they are separate code paths and can drift apart.
TEST(FR_PACK, constOverloadAgreesWhenNothingIsConstant) {
    const uint64_t n = NROWS_PACK;
    auto a = seq(n, 3);
    auto b = seq(n, 800);
    std::vector<FrEl> plain(n), viaConst(n);

    for (uint64_t op = 0; op <= 3; op++) {
        PilFflonk::op_pack(E, n, op, plain.data(), a.data(), b.data());
        PilFflonk::op_pack(E, n, op, viaConst.data(), a.data(), false, b.data(), false);
        for (uint64_t i = 0; i < n; i++) {
            ASSERT_TRUE(E.fr.eq(plain[i], viaConst[i])) << "overloads disagree, op " << op << " row " << i;
        }
    }
}

TEST(FR_PACK, rejectsUnknownOpcode) {
    const uint64_t n = 4;
    auto a = seq(n, 1);
    auto b = seq(n, 2);
    std::vector<FrEl> c(n);

    EXPECT_ANY_THROW(PilFflonk::op_pack(E, n, 4, c.data(), a.data(), b.data()));
    EXPECT_ANY_THROW(PilFflonk::op_pack(E, n, 4, c.data(), a.data(), true, b.data(), false));
}

// In-place operation is used by the evaluator when a temporary is reused as
// both an operand and the destination.
TEST(FR_PACK, worksInPlace) {
    const uint64_t n = 32;
    auto a = seq(n, 5);
    auto b = seq(n, 60);

    std::vector<FrEl> expected(n);
    PilFflonk::op_pack(E, n, (uint64_t)FrOp::mul, expected.data(), a.data(), b.data());

    std::vector<FrEl> inplace(a);
    PilFflonk::op_pack(E, n, (uint64_t)FrOp::mul, inplace.data(), inplace.data(), b.data());

    for (uint64_t i = 0; i < n; i++) {
        ASSERT_TRUE(E.fr.eq(inplace[i], expected[i])) << "in-place mul mismatch at row " << i;
    }
}
