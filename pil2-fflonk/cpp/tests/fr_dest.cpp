// Writeback tests: inversion, combination and the strided store.
#include <gtest/gtest.h>
#include <vector>

#include "fr_dest.hpp"

using PilFflonk::Dest;
using PilFflonk::FrEl;

static AltBn128::Engine &E = AltBn128::Engine::engine;

namespace {

std::vector<FrEl> values(const std::vector<uint64_t> &xs) {
    std::vector<FrEl> v;
    v.reserve(xs.size());
    for (uint64_t x : xs) v.push_back(E.fr.set(x));
    return v;
}

uint64_t asUI(const FrEl &e) { return std::stoull(E.fr.toString(e)); }

} // namespace

// Batch inversion must agree with inverting each value on its own -- they are
// different algorithms and the batch one is easy to get subtly wrong.
TEST(FR_DEST, batchInversionMatchesIndividualInversion) {
    auto batch = values({3, 5, 7, 11, 13, 17, 19, 23});
    auto individual = batch;

    PilFflonk::invertValues(E, batch.data(), batch.size(), /*batch=*/true);
    PilFflonk::invertValues(E, individual.data(), individual.size(), /*batch=*/false);

    for (size_t i = 0; i < batch.size(); i++) {
        ASSERT_TRUE(E.fr.eq(batch[i], individual[i])) << "index " << i;
    }
}

TEST(FR_DEST, inversionIsCorrect) {
    auto v = values({2, 9, 100});
    auto original = v;

    PilFflonk::invertValues(E, v.data(), v.size(), true);

    for (size_t i = 0; i < v.size(); i++) {
        ASSERT_TRUE(E.fr.eq(E.fr.mul(v[i], original[i]), E.fr.one())) << "x * x^-1 != 1 at " << i;
    }
}

// Batch inversion divides by a running product, so a zero anywhere would make
// the whole block wrong rather than just that entry. Refuse instead.
TEST(FR_DEST, rejectsInvertingZero) {
    auto v = values({4, 0, 6});
    EXPECT_ANY_THROW(PilFflonk::invertValues(E, v.data(), v.size(), true));
    EXPECT_ANY_THROW(PilFflonk::invertValues(E, v.data(), v.size(), false));
}

TEST(FR_DEST, invertsASingleValue) {
    auto v = values({7});
    PilFflonk::invertValues(E, v.data(), 1, true);
    ASSERT_TRUE(E.fr.eq(E.fr.mul(v[0], E.fr.set(7)), E.fr.one()));
}

TEST(FR_DEST, multipliesTwoBlocks) {
    auto a = values({2, 3, 4, 5});
    auto b = values({10, 20, 30, 40});

    PilFflonk::multiplyValues(E, a.data(), false, b.data(), false, a.size());

    EXPECT_EQ(asUI(a[0]), 20u);
    EXPECT_EQ(asUI(a[1]), 60u);
    EXPECT_EQ(asUI(a[2]), 120u);
    EXPECT_EQ(asUI(a[3]), 200u);
}

TEST(FR_DEST, multipliesByABroadcastValue) {
    auto a = values({2, 3, 4, 5});
    auto b = values({7});

    PilFflonk::multiplyValues(E, a.data(), false, b.data(), /*constB=*/true, a.size());

    EXPECT_EQ(asUI(a[0]), 14u);
    EXPECT_EQ(asUI(a[3]), 35u);
}

TEST(FR_DEST, storesTightlyPackedByDefault) {
    std::vector<FrEl> buffer(8, E.fr.zero());
    Dest dest{buffer.data(), /*offset=*/0, /*domainSize=*/8};

    auto vals = values({11, 22, 33});
    PilFflonk::storeValues(dest, vals.data(), /*row=*/2, /*nrowsPack=*/3, false);

    EXPECT_EQ(asUI(buffer[2]), 11u);
    EXPECT_EQ(asUI(buffer[3]), 22u);
    EXPECT_EQ(asUI(buffer[4]), 33u);
    EXPECT_EQ(asUI(buffer[1]), 0u) << "must not write before the row";
    EXPECT_EQ(asUI(buffer[5]), 0u) << "must not write past the block";
}

// A non-zero offset interleaves the result into a wider buffer, which is how a
// single column is written into a multi-column trace.
TEST(FR_DEST, honoursTheStride) {
    const uint64_t stride = 3;
    std::vector<FrEl> buffer(4 * stride, E.fr.zero());
    Dest dest{buffer.data(), stride, 4};

    auto vals = values({5, 6, 7, 8});
    PilFflonk::storeValues(dest, vals.data(), 0, 4, false);

    for (uint64_t r = 0; r < 4; r++) {
        EXPECT_EQ(asUI(buffer[r * stride]), 5u + r) << "row " << r;
        // Neighbouring columns must be untouched.
        EXPECT_EQ(asUI(buffer[r * stride + 1]), 0u) << "row " << r << " column 1";
        EXPECT_EQ(asUI(buffer[r * stride + 2]), 0u) << "row " << r << " column 2";
    }
}

TEST(FR_DEST, broadcastsAConstantResult) {
    std::vector<FrEl> buffer(4, E.fr.zero());
    Dest dest{buffer.data(), 0, 4};

    auto vals = values({99, 0, 0, 0});
    PilFflonk::storeValues(dest, vals.data(), 0, 4, /*isConstant=*/true);

    for (uint64_t r = 0; r < 4; r++) EXPECT_EQ(asUI(buffer[r]), 99u) << "row " << r;
}

// Writing past the domain would corrupt whatever follows the buffer; the
// destination knows its own extent, so it can say so.
TEST(FR_DEST, refusesToWritePastTheDomain) {
    std::vector<FrEl> buffer(4, E.fr.zero());
    Dest dest{buffer.data(), 0, 4};
    auto vals = values({1, 2, 3, 4});

    EXPECT_ANY_THROW(PilFflonk::storeValues(dest, vals.data(), /*row=*/2, /*nrowsPack=*/4, false));
}

TEST(FR_DEST, rejectsAnUnboundDestination) {
    Dest dest;
    auto vals = values({1});
    EXPECT_ANY_THROW(PilFflonk::storeValues(dest, vals.data(), 0, 1, false));
}
