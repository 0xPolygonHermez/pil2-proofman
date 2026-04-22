#include "test_helpers.hpp"
#include "../src/goldilocks_cubic_extension.hpp"
#include "../src/goldilocks_cubic_extension_pack.hpp"

TEST(GOLDILOCKS_CUBIC_TEST, one)
{
    uint64_t a[3] = {1, 1, 1};
    int32_t b[3] = {1, 1, 1};
    std::string c[3] = {"92233720347072921606", "92233720347072921606", "92233720347072921606"}; // GOLDILOCKS_PRIME * 5 + 1
    uint64_t d[3] = {1 + GOLDILOCKS_PRIME, 1 + GOLDILOCKS_PRIME, 1 + GOLDILOCKS_PRIME};

    Goldilocks3::Element ina1;
    Goldilocks3::Element ina2;
    Goldilocks3::Element ina3;
    Goldilocks3::Element inb1;
    Goldilocks3::Element inc1;

    Goldilocks3::fromU64(ina1, a);
    Goldilocks3::fromS32(ina2, b);
    Goldilocks3::fromString(ina3, c);
    Goldilocks3::fromU64(inb1, d);
    Goldilocks3::fromString(inc1, c);

    uint64_t ina1_res[3];
    uint64_t ina2_res[3];
    uint64_t ina3_res[3];
    uint64_t inb1_res[3];
    uint64_t inc1_res[3];

    Goldilocks3::toU64(ina1_res, ina1);
    Goldilocks3::toU64(ina2_res, ina2);
    Goldilocks3::toU64(ina3_res, ina3);
    Goldilocks3::toU64(inb1_res, inb1);
    Goldilocks3::toU64(inc1_res, inc1);

    ASSERT_EQ(ina1_res[0], a[0]);
    ASSERT_EQ(ina1_res[1], a[1]);
    ASSERT_EQ(ina1_res[2], a[2]);

    ASSERT_EQ(ina2_res[0], a[0]);
    ASSERT_EQ(ina2_res[1], a[1]);
    ASSERT_EQ(ina2_res[2], a[2]);

    ASSERT_EQ(ina3_res[0], a[0]);
    ASSERT_EQ(ina3_res[1], a[1]);
    ASSERT_EQ(ina3_res[2], a[2]);

    ASSERT_EQ(inb1_res[0], a[0]);
    ASSERT_EQ(inb1_res[1], a[1]);
    ASSERT_EQ(inb1_res[2], a[2]);

    ASSERT_EQ(inc1_res[0], a[0]);
    ASSERT_EQ(inc1_res[1], a[1]);
    ASSERT_EQ(inc1_res[2], a[2]);
}

TEST(GOLDILOCKS_CUBIC_TEST, mul_pack_const_variants_match_row_mul)
{
    constexpr uint64_t nrows = 17;
    std::vector<Goldilocks::Element> a(3 * nrows);
    std::vector<Goldilocks::Element> b(3 * nrows);
    std::vector<Goldilocks::Element> c(3 * nrows);

    for (uint64_t i = 0; i < 3 * nrows; ++i) {
        a[i].fe = (i + 1) * 17;
        b[i].fe = (i + 3) * 29;
    }

    auto check = [&](bool const_a, bool const_b) {
        Goldilocks3::op_pack(nrows, 2, c.data(), a.data(), const_a, b.data(), const_b);
        for (uint64_t i = 0; i < nrows; ++i) {
            Goldilocks3::Element ai = {
                const_a ? a[0] : a[i],
                const_a ? a[1] : a[nrows + i],
                const_a ? a[2] : a[2 * nrows + i],
            };
            Goldilocks3::Element bi = {
                const_b ? b[0] : b[i],
                const_b ? b[1] : b[nrows + i],
                const_b ? b[2] : b[2 * nrows + i],
            };
            Goldilocks3::Element expected;
            Goldilocks3::mul(expected, ai, bi);

            EXPECT_EQ(c[i].fe, expected[0].fe) << "const_a=" << const_a << " const_b=" << const_b << " row=" << i << " limb=0";
            EXPECT_EQ(c[nrows + i].fe, expected[1].fe) << "const_a=" << const_a << " const_b=" << const_b << " row=" << i << " limb=1";
            EXPECT_EQ(c[2 * nrows + i].fe, expected[2].fe) << "const_a=" << const_a << " const_b=" << const_b << " row=" << i << " limb=2";
        }
    };

    check(true, false);
    check(false, true);
    check(true, true);
}
