// Accumulation tests. These pin the arguments pil2 uses in place of pil1's
// plookup h1/h2 and bespoke grand-product Z: std_prod and std_sum are the same
// prefix scan under a different operator.
#include <gtest/gtest.h>
#include <vector>

#include "fr_hints.hpp"

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

TEST(FR_HINTS, prefixSumIsInclusive) {
    auto v = values({1, 2, 3, 4, 5});
    PilFflonk::accOperation(E, v.data(), v.size(), /*add=*/true);

    // Inclusive: entry i holds the sum of inputs 0..i, so the first is
    // untouched and the last is the total.
    EXPECT_EQ(asUI(v[0]), 1u);
    EXPECT_EQ(asUI(v[1]), 3u);
    EXPECT_EQ(asUI(v[2]), 6u);
    EXPECT_EQ(asUI(v[3]), 10u);
    EXPECT_EQ(asUI(v[4]), 15u);
}

TEST(FR_HINTS, prefixProductIsInclusive) {
    auto v = values({2, 3, 4, 5});
    PilFflonk::accOperation(E, v.data(), v.size(), /*add=*/false);

    EXPECT_EQ(asUI(v[0]), 2u);
    EXPECT_EQ(asUI(v[1]), 6u);
    EXPECT_EQ(asUI(v[2]), 24u);
    EXPECT_EQ(asUI(v[3]), 120u);
}

// The property the grand-product argument rests on: the final entry is the
// product over the whole column, so two columns that are permutations of one
// another accumulate to the same total.
TEST(FR_HINTS, grandProductIsPermutationInvariant) {
    auto a = values({7, 11, 13, 17, 19});
    auto b = values({19, 7, 17, 11, 13});  // same multiset, shuffled

    PilFflonk::accOperation(E, a.data(), a.size(), false);
    PilFflonk::accOperation(E, b.data(), b.size(), false);

    ASSERT_TRUE(E.fr.eq(PilFflonk::accTotal(a.data(), a.size()), PilFflonk::accTotal(b.data(), b.size())))
        << "a permutation must not change the grand product";
}

// And it must actually discriminate: a column that is not a permutation gives a
// different total, which is what makes an invalid argument fail.
TEST(FR_HINTS, grandProductDetectsANonPermutation) {
    auto a = values({7, 11, 13, 17, 19});
    auto b = values({7, 11, 13, 17, 23});  // one element differs

    PilFflonk::accOperation(E, a.data(), a.size(), false);
    PilFflonk::accOperation(E, b.data(), b.size(), false);

    ASSERT_FALSE(E.fr.eq(PilFflonk::accTotal(a.data(), a.size()), PilFflonk::accTotal(b.data(), b.size())))
        << "a changed element must change the grand product";
}

// Same for the sum form used by logup.
TEST(FR_HINTS, sumIsPermutationInvariant) {
    auto a = values({4, 8, 15, 16, 23, 42});
    auto b = values({42, 23, 16, 15, 8, 4});

    PilFflonk::accOperation(E, a.data(), a.size(), true);
    PilFflonk::accOperation(E, b.data(), b.size(), true);

    ASSERT_TRUE(E.fr.eq(PilFflonk::accTotal(a.data(), a.size()), PilFflonk::accTotal(b.data(), b.size())));
}

// A zero anywhere collapses the product from that row on -- worth pinning,
// because it is how a badly constructed argument silently degenerates.
TEST(FR_HINTS, aZeroCollapsesTheRunningProduct) {
    auto v = values({5, 6, 0, 7, 8});
    PilFflonk::accOperation(E, v.data(), v.size(), false);

    EXPECT_EQ(asUI(v[1]), 30u);
    EXPECT_EQ(asUI(v[2]), 0u);
    EXPECT_EQ(asUI(v[3]), 0u) << "the product stays zero once a zero is consumed";
    EXPECT_EQ(asUI(v[4]), 0u);
}

TEST(FR_HINTS, wrapsAroundTheFieldModulus) {
    // r - 1 plus 2 must wrap to 1, confirming the scan uses field arithmetic
    // rather than machine integers.
    FrEl negOne = E.fr.neg(E.fr.one());
    std::vector<FrEl> v = {negOne, E.fr.set(2)};

    PilFflonk::accOperation(E, v.data(), v.size(), true);
    ASSERT_TRUE(E.fr.eq(v[1], E.fr.one()));
}

TEST(FR_HINTS, handlesDegenerateLengths) {
    auto one = values({42});
    PilFflonk::accOperation(E, one.data(), 1, true);
    EXPECT_EQ(asUI(one[0]), 42u) << "a single entry is already its own total";

    // Zero length must be a no-op rather than a read of vals[-1].
    auto empty = values({});
    EXPECT_NO_THROW(PilFflonk::accOperation(E, empty.data(), 0, true));

    EXPECT_ANY_THROW(PilFflonk::accTotal(empty.data(), 0));
}

TEST(FR_HINTS, rejectsNullBuffer) {
    EXPECT_ANY_THROW(PilFflonk::accOperation(E, nullptr, 4, true));
}
