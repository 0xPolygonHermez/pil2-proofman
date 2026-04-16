#include "test_helpers.hpp"
#include "../src/goldilocks_cubic_extension.hpp"

#define FFT_SIZE (1 << 4)
#define NUM_REPS 5

TEST(GOLDILOCKS_TEST, one)
{
    uint64_t a = 1;
    uint64_t b = 1 + GOLDILOCKS_PRIME;
    std::string c = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1

    Goldilocks::Element ina1 = Goldilocks::fromU64(a);
    Goldilocks::Element ina2 = Goldilocks::fromS32(a);
    Goldilocks::Element ina3 = Goldilocks::fromString(std::to_string(a));
    Goldilocks::Element inb1 = Goldilocks::fromU64(b);
    Goldilocks::Element inc1 = Goldilocks::fromString(c);

    ASSERT_EQ(Goldilocks::toU64(ina1), a);
    ASSERT_EQ(Goldilocks::toU64(ina2), a);
    ASSERT_EQ(Goldilocks::toU64(ina3), a);
    ASSERT_EQ(Goldilocks::toU64(inb1), a);
    ASSERT_EQ(Goldilocks::toU64(inc1), a);
}

TEST(GOLDILOCKS_TEST, add)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element p_1 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    ASSERT_EQ(Goldilocks::toU64(p_1 + p_1), 2);

    Goldilocks::Element max = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFF);
    ASSERT_EQ(Goldilocks::toU64(max + max), 0X1FFFFFFFC);

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);

    ASSERT_EQ(Goldilocks::toU64(inE1 + inE2), in1 + in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 + inE2 + inE3), in1 + in2 + 1);
    ASSERT_EQ(Goldilocks::toU64(inE1 + inE2 + inE3 + inE4), 1);

    // Edge case (double carry)
    Goldilocks::Element a1 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element a2 = Goldilocks::fromU64(0xFFFFFFFF);
    Goldilocks::Element b1 = (a1 + a2);
    Goldilocks::Element b2 = (b1 + b1);
    ASSERT_EQ(Goldilocks::toU64(b2), 8589934588);
}
#ifdef __AVX2__
TEST(GOLDILOCKS_TEST, add_avx)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element p_1 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    Goldilocks::Element max = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFFULL);

    Goldilocks::Element a1 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element a2 = Goldilocks::fromU64(0xFFFFFFFF);

    Goldilocks::Element *a = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *b = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *c = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));

    a[0] = p_1;
    a[1] = a1;
    a[2] = inE1;
    a[3] = max;

    b[0] = p_1;
    b[1] = a2;
    b[2] = inE2;
    b[3] = max;

    __m256i a_;
    __m256i b_;
    __m256i c_;

    Goldilocks::load_avx(a_, a);
    Goldilocks::set_avx(b_, b[0], b[1], b[2], b[3]); // equivalent to load
    Goldilocks::add_avx(c_, a_, b_);
    Goldilocks::store_avx(c, c_);

    ASSERT_EQ(Goldilocks::toU64(a[0] + b[0]), Goldilocks::toU64(c[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] + b[1]), Goldilocks::toU64(c[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] + b[2]), Goldilocks::toU64(c[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] + b[3]), Goldilocks::toU64(c[3]));

    a[0] = inE3;
    a[1] = c[1];
    a[2] = inE4;
    a[3] = max;

    Goldilocks::load_avx(a_, a);
    Goldilocks::add_avx(b_, a_, c_);
    Goldilocks::store_avx(b, b_);

    ASSERT_EQ(Goldilocks::toU64(a[0] + c[0]), Goldilocks::toU64(b[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] + c[1]), Goldilocks::toU64(b[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] + c[2]), Goldilocks::toU64(b[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] + c[3]), Goldilocks::toU64(b[3]));

    free(a);
    free(b);
    free(c);
}
#endif
#ifdef __AVX512__
TEST(GOLDILOCKS_TEST, add_avx512)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element p_1 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    Goldilocks::Element max = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFFULL);
    Goldilocks::Element a1 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element a2 = Goldilocks::fromU64(0xFFFFFFFF);

    Goldilocks::Element *a = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *b = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *c = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));

    a[0] = p_1;
    a[1] = a1;
    a[2] = inE1;
    a[3] = max;
    a[4] = max;
    a[5] = inE4;
    a[6] = inE1;
    a[7] = inE3;

    b[0] = p_1;
    b[1] = a2;
    b[2] = inE2;
    b[3] = max;
    b[4] = inE1;
    b[5] = inE2;
    b[6] = inE3;
    b[7] = inE4;

    __m512i a_;
    __m512i b_;
    __m512i c_;

    Goldilocks::load_avx512(a_, a);
    Goldilocks::load_avx512(b_, b);
    Goldilocks::add_avx512(c_, a_, b_);
    Goldilocks::store_avx512(c, c_);

    ASSERT_EQ(Goldilocks::toU64(a[0] + b[0]), Goldilocks::toU64(c[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] + b[1]), Goldilocks::toU64(c[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] + b[2]), Goldilocks::toU64(c[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] + b[3]), Goldilocks::toU64(c[3]));
    ASSERT_EQ(Goldilocks::toU64(a[4] + b[4]), Goldilocks::toU64(c[4]));
    ASSERT_EQ(Goldilocks::toU64(a[5] + b[5]), Goldilocks::toU64(c[5]));
    ASSERT_EQ(Goldilocks::toU64(a[6] + b[6]), Goldilocks::toU64(c[6]));
    ASSERT_EQ(Goldilocks::toU64(a[7] + b[7]), Goldilocks::toU64(c[7]));

    free(a);
    free(b);
    free(c);
}
#endif

TEST(GOLDILOCKS_TEST, sub)
{

    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);

    ASSERT_EQ(Goldilocks::toU64(inE1 - inE2), GOLDILOCKS_PRIME + in1 - in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 - inE2 - inE3), GOLDILOCKS_PRIME + in1 - in2 - 1);
    ASSERT_EQ(Goldilocks::toU64(inE1 - inE2 - inE3 - inE4), 5);

    Goldilocks::Element a1 = Goldilocks::fromU64(0xFFFFFFFF00000000LL);
    Goldilocks::Element a2 = Goldilocks::fromU64(0xFFFFFFFFLL);

    Goldilocks::Element a3 = (a1 + a2);
    Goldilocks::Element b2 = Goldilocks::zero() - a3;
    ASSERT_EQ(Goldilocks::toU64(b2), 0XFFFFFFFE00000003LL);
}
#ifdef __AVX2__
TEST(GOLDILOCKS_TEST, sub_avx)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element p_1 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    Goldilocks::Element max = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFFULL);
    Goldilocks::Element a1 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element a2 = Goldilocks::fromU64(0xFFFFFFFF);

    Goldilocks::Element *a = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *b = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *c = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));

    a[0] = inE1;
    a[1] = inE2;
    a[2] = Goldilocks::zero();
    a[3] = a1;

    b[0] = inE1;
    b[1] = inE3;
    b[2] = inE4;
    b[3] = max;

    __m256i a_;
    __m256i b_;
    __m256i c_;

    Goldilocks::load_avx(a_, a);
    Goldilocks::set_avx(b_, b[0], b[1], b[2], b[3]); // equivalent to load
    Goldilocks::sub_avx(c_, a_, b_);
    Goldilocks::store_avx(c, c_);

    ASSERT_EQ(Goldilocks::toU64(a[0] - b[0]), Goldilocks::toU64(c[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] - b[1]), Goldilocks::toU64(c[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] - b[2]), Goldilocks::toU64(c[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] - b[3]), Goldilocks::toU64(c[3]));

    a[0] = p_1;
    a[1] = a2;
    a[2] = Goldilocks::zero();
    a[3] = max;

    Goldilocks::load_avx(a_, a);
    Goldilocks::sub_avx(b_, a_, c_);
    Goldilocks::store_avx(b, b_);

    ASSERT_EQ(Goldilocks::toU64(a[0] - c[0]), Goldilocks::toU64(b[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] - c[1]), Goldilocks::toU64(b[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] - c[2]), Goldilocks::toU64(b[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] - c[3]), Goldilocks::toU64(b[3]));

    // edge case:
    Goldilocks::Element a0 = Goldilocks::fromU64(1);
    Goldilocks::Element b0 = Goldilocks::fromString("6824165416642549846");
    Goldilocks::Element b1 = Goldilocks::fromString("13754891152847927955");
    Goldilocks::Element b2 = Goldilocks::fromString("17916068787382203463");
    Goldilocks::Element b3 = Goldilocks::fromU64(18446744071248801682ULL);

    a[0] = a0;
    a[1] = a0;
    a[2] = a0;
    a[3] = a0;

    b[0] = b0;
    b[1] = b1;
    b[2] = b2;
    b[3] = b3;

    Goldilocks::load_avx(a_, a);
    Goldilocks::load_avx(b_, b);
    Goldilocks::sub_avx(c_, a_, b_);
    Goldilocks::store_avx(c, c_);

    ASSERT_EQ(Goldilocks::toU64(a[0] - b[0]), Goldilocks::toU64(c[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] - b[1]), Goldilocks::toU64(c[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] - b[2]), Goldilocks::toU64(c[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] - b[3]), Goldilocks::toU64(c[3]));

    free(a);
    free(b);
    free(c);
}
#endif
#ifdef __AVX512__
TEST(GOLDILOCKS_TEST, sub_avx512)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element inE5 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    Goldilocks::Element inE6 = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFFULL);
    Goldilocks::Element inE7 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element inE8 = Goldilocks::fromU64(1);
    Goldilocks::Element inE9 = Goldilocks::fromString("6824165416642549846");
    Goldilocks::Element inE10 = Goldilocks::fromString("13754891152847927955");
    Goldilocks::Element inE11 = Goldilocks::fromString("17916068787382203463");
    Goldilocks::Element inE12 = Goldilocks::fromU64(18446744071248801682ULL);

    Goldilocks::Element *a = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *b = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *c = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));

    a[0] = inE1;
    a[1] = inE2;
    a[2] = inE5;
    a[3] = inE7;
    a[4] = inE8;
    a[5] = inE8;
    a[6] = inE8;
    a[7] = inE8;

    b[0] = inE1;
    b[1] = inE3;
    b[2] = inE4;
    b[3] = inE6;
    b[4] = inE9;
    b[5] = inE10;
    b[6] = inE11;
    b[7] = inE12;

    __m512i a_;
    __m512i b_;
    __m512i c_;

    Goldilocks::load_avx512(a_, a);
    Goldilocks::load_avx512(b_, b);
    Goldilocks::sub_avx512(c_, a_, b_);
    Goldilocks::store_avx512(c, c_);

    ASSERT_EQ(Goldilocks::toU64(a[0] - b[0]), Goldilocks::toU64(c[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] - b[1]), Goldilocks::toU64(c[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] - b[2]), Goldilocks::toU64(c[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] - b[3]), Goldilocks::toU64(c[3]));
    ASSERT_EQ(Goldilocks::toU64(a[4] - b[4]), Goldilocks::toU64(c[4]));
    ASSERT_EQ(Goldilocks::toU64(a[5] - b[5]), Goldilocks::toU64(c[5]));
    ASSERT_EQ(Goldilocks::toU64(a[6] - b[6]), Goldilocks::toU64(c[6]));
    ASSERT_EQ(Goldilocks::toU64(a[7] - b[7]), Goldilocks::toU64(c[7]));

    free(a);
    free(b);
    free(c);
}
#endif

TEST(GOLDILOCKS_TEST, mul)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);

    ASSERT_EQ(Goldilocks::toU64(inE1 * inE2), in1 * in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 * inE2 * inE3), in1 * in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 * inE2 * inE3 * inE4), 0XFFFFFFFEFFFFFEBDLL);
}
#ifdef __AVX2__
TEST(GOLDILOCKS_TEST, mul_avx)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element p_1 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    Goldilocks::Element max = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFFULL);
    Goldilocks::Element a1 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element a2 = Goldilocks::fromU64(0xFFFFFFFF);

    Goldilocks::Element *a = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *b = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *c = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));

    a[0] = inE1;
    a[1] = inE2;
    a[2] = Goldilocks::zero();
    a[3] = inE4;

    b[0] = a1;
    b[1] = inE3;
    b[2] = inE4;
    b[3] = p_1;

    __m256i a_;
    __m256i b_;
    __m256i c_;

    Goldilocks::load_avx(a_, a);
    Goldilocks::set_avx(b_, b[0], b[1], b[2], b[3]); // equivalent to load
    Goldilocks::mult_avx(c_, a_, b_);
    Goldilocks::store_avx(c, c_);

    ASSERT_EQ(Goldilocks::toU64(a[0] * b[0]), Goldilocks::toU64(c[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] * b[1]), Goldilocks::toU64(c[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] * b[2]), Goldilocks::toU64(c[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] * b[3]), Goldilocks::toU64(c[3]));

    a[0] = p_1;
    a[1] = a2;
    a[2] = Goldilocks::zero();
    a[3] = max;

    Goldilocks::load_avx(a_, a);
    Goldilocks::mult_avx(b_, a_, c_);
    Goldilocks::store_avx(b, b_);

    ASSERT_EQ(Goldilocks::toU64(a[0] * c[0]), Goldilocks::toU64(b[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] * c[1]), Goldilocks::toU64(b[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] * c[2]), Goldilocks::toU64(b[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] * c[3]), Goldilocks::toU64(b[3]));

    free(a);
    free(b);
    free(c);
}
#endif
#ifdef __AVX512__
TEST(GOLDILOCKS_TEST, mul_avx512)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element inE5 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    Goldilocks::Element inE6 = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFFULL);
    Goldilocks::Element inE7 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element inE8 = Goldilocks::fromU64(1);
    Goldilocks::Element inE9 = Goldilocks::fromString("6824165416642549846");
    Goldilocks::Element inE10 = Goldilocks::fromString("13754891152847927955");
    Goldilocks::Element inE11 = Goldilocks::fromString("17916068787382203463");
    Goldilocks::Element inE12 = Goldilocks::fromU64(18446744071248801682ULL);
    Goldilocks::Element inE13 = Goldilocks::zero();

    Goldilocks::Element *a = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *b = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *c = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));

    a[0] = inE1;
    a[1] = inE2;
    a[2] = inE13;
    a[3] = inE4;
    a[4] = inE9;
    a[5] = inE10;
    a[6] = inE11;
    a[7] = inE12;

    b[0] = inE5;
    b[1] = inE3;
    b[2] = inE4;
    b[3] = inE6;
    b[4] = inE7;
    b[5] = inE3;
    b[6] = inE4;
    b[7] = inE8;

    __m512i a_;
    __m512i b_;
    __m512i c_;

    Goldilocks::load_avx512(a_, a);
    Goldilocks::load_avx512(b_, b);
    Goldilocks::mult_avx512(c_, a_, b_);
    Goldilocks::store_avx512(c, c_);

    ASSERT_EQ(Goldilocks::toU64(a[0] * b[0]), Goldilocks::toU64(c[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] * b[1]), Goldilocks::toU64(c[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] * b[2]), Goldilocks::toU64(c[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] * b[3]), Goldilocks::toU64(c[3]));
    ASSERT_EQ(Goldilocks::toU64(a[4] * b[4]), Goldilocks::toU64(c[4]));
    ASSERT_EQ(Goldilocks::toU64(a[5] * b[5]), Goldilocks::toU64(c[5]));
    ASSERT_EQ(Goldilocks::toU64(a[6] * b[6]), Goldilocks::toU64(c[6]));
    ASSERT_EQ(Goldilocks::toU64(a[7] * b[7]), Goldilocks::toU64(c[7]));

    free(a);
    free(b);
    free(c);
}
#endif
#ifdef __AVX2__
TEST(GOLDILOCKS_TEST, mul_avx_8)
{
    int32_t in1 = 3;
    int32_t in2 = 9;
    int32_t in3 = 9;
    int32_t in4 = 100;
    int32_t in5 = 3;
    int32_t in6 = 9;
    int32_t in7 = 9;
    int32_t in8 = 100;

    Goldilocks::Element inE1 = Goldilocks::fromS32(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromS32(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element inE5 = Goldilocks::fromS32(in5);
    Goldilocks::Element inE6 = Goldilocks::fromS32(in6);
    Goldilocks::Element inE7 = Goldilocks::fromS32(in7);
    Goldilocks::Element inE8 = Goldilocks::fromS32(in8);

    Goldilocks::Element *a = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *b = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *c = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));

    a[0] = inE1;
    a[1] = inE2;
    a[2] = inE3;
    a[3] = inE4;

    b[0] = inE5;
    b[1] = inE6;
    b[2] = inE7;
    b[3] = inE8;

    __m256i a_;
    __m256i b_;
    __m256i c_;

    Goldilocks::load_avx(a_, a);
    Goldilocks::set_avx(b_, b[0], b[1], b[2], b[3]); // equivalent to load
    Goldilocks::mult_avx_8(c_, a_, b_);
    Goldilocks::store_avx(c, c_);

    ASSERT_EQ(Goldilocks::toU64(a[0] * b[0]), Goldilocks::toU64(c[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] * b[1]), Goldilocks::toU64(c[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] * b[2]), Goldilocks::toU64(c[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] * b[3]), Goldilocks::toU64(c[3]));

    free(a);
    free(b);
    free(c);
}
#endif
#ifdef __AVX512__
TEST(GOLDILOCKS_TEST, mul_avx512_8)
{
    int32_t in1 = 3;
    int32_t in2 = 9;
    int32_t in3 = 9;
    int32_t in4 = 100;
    int32_t in5 = 0;
    int32_t in6 = 1;
    int32_t in7 = 64;
    int32_t in8 = 2;

    Goldilocks::Element inE1 = Goldilocks::fromS32(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromS32(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element inE5 = Goldilocks::fromS32(in5);
    Goldilocks::Element inE6 = Goldilocks::fromS32(in6);
    Goldilocks::Element inE7 = Goldilocks::fromS32(in7);
    Goldilocks::Element inE8 = Goldilocks::fromS32(in8);

    Goldilocks::Element *a = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *b = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *c = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));

    a[0] = inE1;
    a[1] = inE2;
    a[2] = inE3;
    a[3] = inE4;
    a[4] = inE5;
    a[5] = inE7;
    a[6] = inE8;
    a[7] = inE2;

    b[0] = inE5;
    b[1] = inE6;
    b[2] = inE7;
    b[3] = inE8;
    b[4] = inE6;
    b[5] = inE1;
    b[6] = inE2;
    b[7] = inE3;

    __m512i a_;
    __m512i b_;
    __m512i c_;

    Goldilocks::load_avx512(a_, a);
    Goldilocks::load_avx512(b_, b);
    Goldilocks::mult_avx512_8(c_, a_, b_);
    Goldilocks::store_avx512(c, c_);

    ASSERT_EQ(Goldilocks::toU64(a[0] * b[0]), Goldilocks::toU64(c[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] * b[1]), Goldilocks::toU64(c[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] * b[2]), Goldilocks::toU64(c[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] * b[3]), Goldilocks::toU64(c[3]));
    ASSERT_EQ(Goldilocks::toU64(a[4] * b[4]), Goldilocks::toU64(c[4]));
    ASSERT_EQ(Goldilocks::toU64(a[5] * b[5]), Goldilocks::toU64(c[5]));
    ASSERT_EQ(Goldilocks::toU64(a[6] * b[6]), Goldilocks::toU64(c[6]));
    ASSERT_EQ(Goldilocks::toU64(a[7] * b[7]), Goldilocks::toU64(c[7]));

    free(a);
    free(b);
    free(c);
}
#endif
#ifdef __AVX2__
TEST(GOLDILOCKS_TEST, square_avx)
{
    uint64_t in1 = 3;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element a1 = Goldilocks::fromU64(0xFFFFFFFF00000000);

    Goldilocks::Element *a = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *c = (Goldilocks::Element *)malloc(4 * (sizeof(Goldilocks::Element)));

    a[0] = inE1;
    a[1] = inE3;
    a[2] = Goldilocks::zero();
    a[3] = a1;

    __m256i a_;
    __m256i c_;

    Goldilocks::load_avx(a_, a);
    Goldilocks::square_avx(c_, a_);
    Goldilocks::store_avx(c, c_);

    ASSERT_EQ(Goldilocks::toU64(a[0] * a[0]), Goldilocks::toU64(c[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] * a[1]), Goldilocks::toU64(c[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] * a[2]), Goldilocks::toU64(c[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] * a[3]), Goldilocks::toU64(c[3]));

    Goldilocks::square_avx(a_, c_);
    Goldilocks::store_avx(a, a_);

    ASSERT_EQ(Goldilocks::toU64(c[0] * c[0]), Goldilocks::toU64(a[0]));
    ASSERT_EQ(Goldilocks::toU64(c[1] * c[1]), Goldilocks::toU64(a[1]));
    ASSERT_EQ(Goldilocks::toU64(c[2] * c[2]), Goldilocks::toU64(a[2]));
    ASSERT_EQ(Goldilocks::toU64(c[3] * c[3]), Goldilocks::toU64(a[3]));

    free(a);
    free(c);
}
#endif
#ifdef __AVX512__
TEST(GOLDILOCKS_TEST, square_avx512)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element inE5 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    Goldilocks::Element inE6 = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFFULL);
    Goldilocks::Element inE7 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element inE8 = Goldilocks::fromU64(1);
    Goldilocks::Element inE9 = Goldilocks::fromString("6824165416642549846");
    Goldilocks::Element inE10 = Goldilocks::fromString("13754891152847927955");
    Goldilocks::Element inE11 = Goldilocks::fromString("17916068787382203463");
    Goldilocks::Element inE12 = Goldilocks::fromU64(18446744071248801682ULL);
    Goldilocks::Element inE13 = Goldilocks::zero();

    Goldilocks::Element *a = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *b = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));
    Goldilocks::Element *c = (Goldilocks::Element *)malloc(8 * (sizeof(Goldilocks::Element)));

    a[0] = inE1;
    a[1] = inE2;
    a[2] = inE13;
    a[3] = inE4;
    a[4] = inE9;
    a[5] = inE10;
    a[6] = inE11;
    a[7] = inE12;

    b[0] = inE5;
    b[1] = inE3;
    b[2] = inE4;
    b[3] = inE6;
    b[4] = inE7;
    b[5] = inE3;
    b[6] = inE4;
    b[7] = inE8;

    __m512i a_;
    __m512i b_;
    __m512i c_;

    Goldilocks::load_avx512(a_, a);
    Goldilocks::square_avx512(c_, a_);
    Goldilocks::store_avx512(c, c_);

    ASSERT_EQ(Goldilocks::toU64(a[0] * a[0]), Goldilocks::toU64(c[0]));
    ASSERT_EQ(Goldilocks::toU64(a[1] * a[1]), Goldilocks::toU64(c[1]));
    ASSERT_EQ(Goldilocks::toU64(a[2] * a[2]), Goldilocks::toU64(c[2]));
    ASSERT_EQ(Goldilocks::toU64(a[3] * a[3]), Goldilocks::toU64(c[3]));
    ASSERT_EQ(Goldilocks::toU64(a[4] * a[4]), Goldilocks::toU64(c[4]));
    ASSERT_EQ(Goldilocks::toU64(a[5] * a[5]), Goldilocks::toU64(c[5]));
    ASSERT_EQ(Goldilocks::toU64(a[6] * a[6]), Goldilocks::toU64(c[6]));
    ASSERT_EQ(Goldilocks::toU64(a[7] * a[7]), Goldilocks::toU64(c[7]));

    Goldilocks::load_avx512(b_, b);
    Goldilocks::square_avx512(c_, b_);
    Goldilocks::store_avx512(c, c_);

    ASSERT_EQ(Goldilocks::toU64(b[0] * b[0]), Goldilocks::toU64(c[0]));
    ASSERT_EQ(Goldilocks::toU64(b[1] * b[1]), Goldilocks::toU64(c[1]));
    ASSERT_EQ(Goldilocks::toU64(b[2] * b[2]), Goldilocks::toU64(c[2]));
    ASSERT_EQ(Goldilocks::toU64(b[3] * b[3]), Goldilocks::toU64(c[3]));
    ASSERT_EQ(Goldilocks::toU64(b[4] * b[4]), Goldilocks::toU64(c[4]));
    ASSERT_EQ(Goldilocks::toU64(b[5] * b[5]), Goldilocks::toU64(c[5]));
    ASSERT_EQ(Goldilocks::toU64(b[6] * b[6]), Goldilocks::toU64(c[6]));
    ASSERT_EQ(Goldilocks::toU64(b[7] * b[7]), Goldilocks::toU64(c[7]));

    free(a);
    free(b);
    free(c);
}
#endif
#ifdef __AVX2__
TEST(GOLDILOCKS_TEST, dot_avx)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element p_1 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    Goldilocks::Element max = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFFULL);
    Goldilocks::Element a1 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element a2 = Goldilocks::fromU64(0xFFFFFFFF);

    Goldilocks::Element *a = (Goldilocks::Element *)aligned_alloc(32, 12 * sizeof(Goldilocks::Element));
    Goldilocks::Element *b = (Goldilocks::Element *)aligned_alloc(32, 12 * sizeof(Goldilocks::Element));

    a[0] = inE1;
    a[1] = inE2;
    a[2] = Goldilocks::one();
    a[3] = inE3;
    a[4] = inE4;
    a[5] = max;
    a[6] = a1;
    a[7] = a2;
    a[8] = max * max;
    a[9] = p_1;
    a[10] = a1 * a1;
    a[11] = inE4 * p_1;

    b[0] = max;
    b[1] = a1;
    b[2] = inE4;
    b[3] = p_1;
    b[4] = Goldilocks::zero();
    b[5] = inE3;
    b[6] = inE1;
    b[7] = (a1 * inE1);
    b[8] = max;
    b[9] = inE4;
    b[10] = p_1;
    b[11] = Goldilocks::one();

    Goldilocks::Element dotp1 = Goldilocks::zero();
    for (int i = 0; i < 12; ++i)
    {
        dotp1 = dotp1 + a[i] * b[i];
    }

    __m256i a0_;
    __m256i a1_;
    __m256i a2_;

    Goldilocks::load_avx_a(a0_, &(a[0]));
    Goldilocks::load_avx_a(a1_, &(a[4]));
    Goldilocks::load_avx_a(a2_, &(a[8]));

    Goldilocks::Element dotp2 = Goldilocks::dot_avx(a0_, a1_, a2_, b);
    ASSERT_EQ(Goldilocks::toU64(dotp1), Goldilocks::toU64(dotp2));
    free(a);
    free(b);
}
#endif
#ifdef __AVX512__
TEST(GOLDILOCKS_TEST, dot_avx512)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element p_1 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    Goldilocks::Element max = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFFULL);
    Goldilocks::Element a1 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element a2 = Goldilocks::fromU64(0xFFFFFFFF);

    Goldilocks::Element *a = (Goldilocks::Element *)aligned_alloc(64, 24 * sizeof(Goldilocks::Element));
    Goldilocks::Element *b = (Goldilocks::Element *)aligned_alloc(64, 12 * sizeof(Goldilocks::Element));

    a[0] = inE1;
    a[1] = inE2;
    a[2] = Goldilocks::one();
    a[3] = inE3;
    a[4] = inE1;
    a[5] = inE2;
    a[6] = Goldilocks::one();
    a[7] = inE3;
    a[8] = inE4;
    a[9] = max;
    a[10] = a1;
    a[11] = a2;
    a[12] = inE4;
    a[13] = max;
    a[14] = a1;
    a[15] = a2;
    a[16] = max * max;
    a[17] = p_1;
    a[18] = a1 * a1;
    a[19] = inE4 * p_1;
    a[20] = max * max;
    a[21] = p_1;
    a[22] = a1 * a1;
    a[23] = inE4 * p_1;

    b[0] = max;
    b[1] = a1;
    b[2] = inE4;
    b[3] = p_1;
    b[4] = Goldilocks::zero();
    b[5] = inE3;
    b[6] = inE1;
    b[7] = (a1 * inE1);
    b[8] = max;
    b[9] = inE4;
    b[10] = p_1;
    b[11] = Goldilocks::one();

    Goldilocks::Element dotp1 = Goldilocks::zero();
    for (int k = 0; k < 3; k += 1)
    {
        for (int i = 0; i < 4; ++i)
        {
            dotp1 = dotp1 + a[k * 8 + i] * b[k * 4 + i];
        }
    }

    __m512i a0_;
    __m512i a1_;
    __m512i a2_;

    // Not aligned
    Goldilocks::load_avx512(a0_, &(a[0]));
    Goldilocks::load_avx512(a1_, &(a[8]));
    Goldilocks::load_avx512(a2_, &(a[16]));

    Goldilocks::Element dotp2[2];

    Goldilocks::dot_avx512(dotp2, a0_, a1_, a2_, b);
    ASSERT_EQ(Goldilocks::toU64(dotp2[0]), Goldilocks::toU64(dotp2[1]));
    ASSERT_EQ(Goldilocks::toU64(dotp1), Goldilocks::toU64(dotp2[0]));

    free(a);
    free(b);
}
#endif
#ifdef __AVX2__
/*TEST(GOLDILOCKS_TEST, mult_avx_4x12)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element p_1 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    Goldilocks::Element max = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFFULL);
    Goldilocks::Element a1 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element a2 = Goldilocks::fromU64(0xFFFFFFFF);

    Goldilocks::Element *a = (Goldilocks::Element *)aligned_alloc(32, 12 * sizeof(Goldilocks::Element));
    Goldilocks::Element *Mat = (Goldilocks::Element *)aligned_alloc(32, 48 * sizeof(Goldilocks::Element));
    Goldilocks::Element *b1 = (Goldilocks::Element *)aligned_alloc(32, 4 * sizeof(Goldilocks::Element));
    Goldilocks::Element *b2 = (Goldilocks::Element *)aligned_alloc(32, 4 * sizeof(Goldilocks::Element));

    a[0] = inE1;
    a[1] = inE2;
    a[2] = Goldilocks::one();
    a[3] = inE3;
    a[4] = inE4;
    a[5] = max;
    a[6] = a1;
    a[7] = a2;
    a[8] = max * max;
    a[9] = p_1;
    a[10] = a1 * a1;
    a[11] = inE4 * p_1;

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 12; ++j)
        {
            Mat[i * 12 + j] = PoseidonGoldilocksConstants::M[i][j];
        }
    }

    // product
    for (int i = 0; i < 4; ++i)
    {
        Goldilocks::Element sum = Goldilocks::zero();
        for (int j = 0; j < 12; ++j)
        {
            sum = sum + (Mat[i * 12 + j] * a[j]);
        }
        b1[i] = sum;
    }

    // avx product
    __m256i a0_;
    __m256i a1_;
    __m256i a2_;

    Goldilocks::load_avx(a0_, &(a[0]));
    Goldilocks::load_avx(a1_, &(a[4]));
    Goldilocks::load_avx(a2_, &(a[8]));
    __m256i b_;
    Goldilocks::mmult_avx_4x12(b_, a0_, a1_, a2_, &(Mat[0]));
    Goldilocks::store_avx(b2, b_);

    ASSERT_EQ(Goldilocks::toU64(b1[0]), Goldilocks::toU64(b2[0]));
    ASSERT_EQ(Goldilocks::toU64(b1[1]), Goldilocks::toU64(b2[1]));
    ASSERT_EQ(Goldilocks::toU64(b1[2]), Goldilocks::toU64(b2[2]));
    ASSERT_EQ(Goldilocks::toU64(b1[3]), Goldilocks::toU64(b2[3]));
    free(a);
    free(Mat);
    free(b1);
    free(b2);
}*/
#endif
#ifdef __AVX512__
TEST(GOLDILOCKS_TEST, mult_avx512_4x12)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element p_1 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    Goldilocks::Element max = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFFULL);
    Goldilocks::Element a1 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element a2 = Goldilocks::fromU64(0xFFFFFFFF);

    Goldilocks::Element *a = (Goldilocks::Element *)aligned_alloc(64, 24 * sizeof(Goldilocks::Element));
    Goldilocks::Element *Mat = (Goldilocks::Element *)aligned_alloc(64, 48 * sizeof(Goldilocks::Element));
    Goldilocks::Element *b1 = (Goldilocks::Element *)aligned_alloc(64, 8 * sizeof(Goldilocks::Element));
    Goldilocks::Element *b2 = (Goldilocks::Element *)aligned_alloc(64, 8 * sizeof(Goldilocks::Element));

    a[0] = inE1;
    a[1] = inE2;
    a[2] = Goldilocks::one();
    a[3] = inE3;
    a[4] = inE1;
    a[5] = inE2;
    a[6] = Goldilocks::one();
    a[7] = inE3;

    a[8] = inE4;
    a[9] = max;
    a[10] = a1;
    a[11] = a2;
    a[12] = inE4;
    a[13] = max;
    a[14] = a1;
    a[15] = a2;

    a[16] = max * max;
    a[17] = p_1;
    a[18] = a1 * a1;
    a[19] = inE4 * p_1;
    a[20] = max * max;
    a[21] = p_1;
    a[22] = a1 * a1;
    a[23] = inE4 * p_1;

    // Use arbitrary values (PoseidonGoldilocksConstants removed in Poseidon2 refactor)
    for (int k = 0; k < 48; ++k)
        Mat[k] = Goldilocks::fromU64(k + 1);

    // product
    for (int i = 0; i < 4; ++i)
    {
        Goldilocks::Element sum = Goldilocks::zero();
        for (int k = 0; k < 3; ++k)
        {
            for (int j = 0; j < 4; ++j)
            {
                sum = sum + (Mat[i * 12 + k * 4 + j] * a[k * 8 + j]);
            }
        }
        b1[i] = sum;
        b1[i + 4] = sum;
    }

    // avx product
    __m512i a0_;
    __m512i a1_;
    __m512i a2_;

    Goldilocks::load_avx512(a0_, &(a[0]));
    Goldilocks::load_avx512(a1_, &(a[8]));
    Goldilocks::load_avx512(a2_, &(a[16]));
    __m512i b_;
    Goldilocks::mmult_avx512_4x12(b_, a0_, a1_, a2_, &(Mat[0]));
    Goldilocks::store_avx512(b2, b_);

    ASSERT_EQ(Goldilocks::toU64(b1[0]), Goldilocks::toU64(b2[0]));
    ASSERT_EQ(Goldilocks::toU64(b1[1]), Goldilocks::toU64(b2[1]));
    ASSERT_EQ(Goldilocks::toU64(b1[2]), Goldilocks::toU64(b2[2]));
    ASSERT_EQ(Goldilocks::toU64(b1[3]), Goldilocks::toU64(b2[3]));
    ASSERT_EQ(Goldilocks::toU64(b1[4]), Goldilocks::toU64(b2[4]));
    ASSERT_EQ(Goldilocks::toU64(b1[5]), Goldilocks::toU64(b2[5]));
    ASSERT_EQ(Goldilocks::toU64(b1[6]), Goldilocks::toU64(b2[6]));
    ASSERT_EQ(Goldilocks::toU64(b1[7]), Goldilocks::toU64(b2[7]));

    // avx product small coeficients
    for (int i = 0; i < 48; ++i)
    {
        Mat[i].fe = Mat[i].fe % 256;
    }
    for (int i = 0; i < 4; ++i)
    {
        Goldilocks::Element sum = Goldilocks::zero();
        for (int k = 0; k < 3; ++k)
        {
            for (int j = 0; j < 4; ++j)
            {
                sum = sum + (Mat[i * 12 + k * 4 + j] * a[k * 8 + j]);
            }
        }
        b1[i] = sum;
        b1[i + 4] = sum;
    }

    Goldilocks::mmult_avx512_4x12_8(b_, a0_, a1_, a2_, &(Mat[0]));
    Goldilocks::store_avx512(b2, b_);

    ASSERT_EQ(Goldilocks::toU64(b1[0]), Goldilocks::toU64(b2[0]));
    ASSERT_EQ(Goldilocks::toU64(b1[1]), Goldilocks::toU64(b2[1]));
    ASSERT_EQ(Goldilocks::toU64(b1[2]), Goldilocks::toU64(b2[2]));
    ASSERT_EQ(Goldilocks::toU64(b1[3]), Goldilocks::toU64(b2[3]));
    ASSERT_EQ(Goldilocks::toU64(b2[4]), Goldilocks::toU64(b2[4]));
    ASSERT_EQ(Goldilocks::toU64(b2[5]), Goldilocks::toU64(b2[5]));
    ASSERT_EQ(Goldilocks::toU64(b2[6]), Goldilocks::toU64(b2[6]));
    ASSERT_EQ(Goldilocks::toU64(b2[7]), Goldilocks::toU64(b2[7]));

    free(a);
    free(Mat);
    free(b1);
    free(b2);
}
#endif
#ifdef __AVX2__
/*TEST(GOLDILOCKS_TEST, mmult_avx)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element p_1 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    Goldilocks::Element max = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFFULL);
    Goldilocks::Element a1 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element a2 = Goldilocks::fromU64(0xFFFFFFFF);

    Goldilocks::Element *a = (Goldilocks::Element *)aligned_alloc(32, 12 * sizeof(Goldilocks::Element));
    Goldilocks::Element *Mat = (Goldilocks::Element *)aligned_alloc(32, 144 * sizeof(Goldilocks::Element));
    Goldilocks::Element *b = (Goldilocks::Element *)aligned_alloc(32, 12 * sizeof(Goldilocks::Element));

    a[0] = inE1;
    a[1] = inE2;
    a[2] = Goldilocks::one();
    a[3] = inE3;
    a[4] = inE4;
    a[5] = max;
    a[6] = a1;
    a[7] = a2;
    a[8] = max * max;
    a[9] = p_1;
    a[10] = a1 * a1;
    a[11] = inE4 * p_1;

    for (int i = 0; i < 12; ++i)
    {
        for (int j = 0; j < 12; ++j)
        {
            Mat[i * 12 + j] = PoseidonGoldilocksConstants::M[i][j];
        }
    }

    // product
    for (int i = 0; i < 12; ++i)
    {
        Goldilocks::Element sum = Goldilocks::zero();
        for (int j = 0; j < 12; ++j)
        {
            sum = sum + (Mat[i * 12 + j] * a[j]);
        }
        b[i] = sum;
    }

    // avx product
    __m256i a0_;
    __m256i a1_;
    __m256i a2_;

    Goldilocks::load_avx(a0_, &(a[0]));
    Goldilocks::load_avx(a1_, &(a[4]));
    Goldilocks::load_avx(a2_, &(a[8]));

    Goldilocks::mmult_avx(a0_, a1_, a2_, &(Mat[0]));

    Goldilocks::store_avx(&(a[0]), a0_);
    Goldilocks::store_avx(&(a[4]), a1_);
    Goldilocks::store_avx(&(a[8]), a2_);

    ASSERT_EQ(Goldilocks::toU64(b[0]), Goldilocks::toU64(a[0]));
    ASSERT_EQ(Goldilocks::toU64(b[1]), Goldilocks::toU64(a[1]));
    ASSERT_EQ(Goldilocks::toU64(b[2]), Goldilocks::toU64(a[2]));
    ASSERT_EQ(Goldilocks::toU64(b[3]), Goldilocks::toU64(a[3]));
    ASSERT_EQ(Goldilocks::toU64(b[4]), Goldilocks::toU64(a[4]));
    ASSERT_EQ(Goldilocks::toU64(b[5]), Goldilocks::toU64(a[5]));
    ASSERT_EQ(Goldilocks::toU64(b[6]), Goldilocks::toU64(a[6]));
    ASSERT_EQ(Goldilocks::toU64(b[7]), Goldilocks::toU64(a[7]));
    ASSERT_EQ(Goldilocks::toU64(b[8]), Goldilocks::toU64(a[8]));
    ASSERT_EQ(Goldilocks::toU64(b[9]), Goldilocks::toU64(a[9]));
    ASSERT_EQ(Goldilocks::toU64(b[10]), Goldilocks::toU64(a[10]));
    ASSERT_EQ(Goldilocks::toU64(b[11]), Goldilocks::toU64(a[11]));

    // avx product aligned
    a[0] = inE1;
    a[1] = inE2;
    a[2] = Goldilocks::one();
    a[3] = inE3;
    a[4] = inE4;
    a[5] = max;
    a[6] = a1;
    a[7] = a2;
    a[8] = max * max;
    a[9] = p_1;
    a[10] = a1 * a1;
    a[11] = inE4 * p_1;
    Goldilocks::load_avx_a(a0_, &(a[0]));
    Goldilocks::load_avx_a(a1_, &(a[4]));
    Goldilocks::load_avx_a(a2_, &(a[8]));
    Goldilocks::mmult_avx_a(a0_, a1_, a2_, &(Mat[0]));
    Goldilocks::store_avx_a(&(a[0]), a0_);
    Goldilocks::store_avx_a(&(a[4]), a1_);
    Goldilocks::store_avx_a(&(a[8]), a2_);

    ASSERT_EQ(Goldilocks::toU64(b[0]), Goldilocks::toU64(a[0]));
    ASSERT_EQ(Goldilocks::toU64(b[1]), Goldilocks::toU64(a[1]));
    ASSERT_EQ(Goldilocks::toU64(b[2]), Goldilocks::toU64(a[2]));
    ASSERT_EQ(Goldilocks::toU64(b[3]), Goldilocks::toU64(a[3]));
    ASSERT_EQ(Goldilocks::toU64(b[4]), Goldilocks::toU64(a[4]));
    ASSERT_EQ(Goldilocks::toU64(b[5]), Goldilocks::toU64(a[5]));
    ASSERT_EQ(Goldilocks::toU64(b[6]), Goldilocks::toU64(a[6]));
    ASSERT_EQ(Goldilocks::toU64(b[7]), Goldilocks::toU64(a[7]));
    ASSERT_EQ(Goldilocks::toU64(b[8]), Goldilocks::toU64(a[8]));
    ASSERT_EQ(Goldilocks::toU64(b[9]), Goldilocks::toU64(a[9]));
    ASSERT_EQ(Goldilocks::toU64(b[10]), Goldilocks::toU64(a[10]));
    ASSERT_EQ(Goldilocks::toU64(b[11]), Goldilocks::toU64(a[11]));

    // avx product_8
    a[0] = inE1;
    a[1] = inE2;
    a[2] = Goldilocks::one();
    a[3] = inE3;
    a[4] = inE4;
    a[5] = max;
    a[6] = a1;
    a[7] = a2;
    a[8] = max * max;
    a[9] = p_1;
    a[10] = a1 * a1;
    a[11] = inE4 * p_1;
    Goldilocks::load_avx(a0_, &(a[0]));
    Goldilocks::load_avx(a1_, &(a[4]));
    Goldilocks::load_avx(a2_, &(a[8]));
    Goldilocks::mmult_avx_8(a0_, a1_, a2_, &(Mat[0]));
    Goldilocks::store_avx(&(a[0]), a0_);
    Goldilocks::store_avx(&(a[4]), a1_);
    Goldilocks::store_avx(&(a[8]), a2_);

    ASSERT_EQ(Goldilocks::toU64(b[0]), Goldilocks::toU64(a[0]));
    ASSERT_EQ(Goldilocks::toU64(b[1]), Goldilocks::toU64(a[1]));
    ASSERT_EQ(Goldilocks::toU64(b[2]), Goldilocks::toU64(a[2]));
    ASSERT_EQ(Goldilocks::toU64(b[3]), Goldilocks::toU64(a[3]));
    ASSERT_EQ(Goldilocks::toU64(b[4]), Goldilocks::toU64(a[4]));
    ASSERT_EQ(Goldilocks::toU64(b[5]), Goldilocks::toU64(a[5]));
    ASSERT_EQ(Goldilocks::toU64(b[6]), Goldilocks::toU64(a[6]));
    ASSERT_EQ(Goldilocks::toU64(b[7]), Goldilocks::toU64(a[7]));
    ASSERT_EQ(Goldilocks::toU64(b[8]), Goldilocks::toU64(a[8]));
    ASSERT_EQ(Goldilocks::toU64(b[9]), Goldilocks::toU64(a[9]));
    ASSERT_EQ(Goldilocks::toU64(b[10]), Goldilocks::toU64(a[10]));
    ASSERT_EQ(Goldilocks::toU64(b[11]), Goldilocks::toU64(a[11]));

    free(a);
    free(Mat);
    free(b);
}*/
#endif
#ifdef __AVX512__
TEST(GOLDILOCKS_TEST, mmult_avx512)
{
    uint64_t in1 = 3;
    int32_t in2 = 9;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element p_1 = Goldilocks::fromU64(0XFFFFFFFF00000002LL);
    Goldilocks::Element max = Goldilocks::fromU64(0XFFFFFFFFFFFFFFFFULL);
    Goldilocks::Element a1 = Goldilocks::fromU64(0xFFFFFFFF00000000);
    Goldilocks::Element a2 = Goldilocks::fromU64(0xFFFFFFFF);

    Goldilocks::Element *a = (Goldilocks::Element *)aligned_alloc(64, 24 * sizeof(Goldilocks::Element));
    Goldilocks::Element *Mat = (Goldilocks::Element *)aligned_alloc(64, 144 * sizeof(Goldilocks::Element));
    Goldilocks::Element *b = (Goldilocks::Element *)aligned_alloc(64, 24 * sizeof(Goldilocks::Element));

    a[0] = inE1;
    a[1] = inE2;
    a[2] = Goldilocks::one();
    a[3] = inE3;
    a[4] = inE1;
    a[5] = inE2;
    a[6] = Goldilocks::one();
    a[7] = inE3;

    a[8] = inE4;
    a[9] = max;
    a[10] = a1;
    a[11] = a2;
    a[12] = inE4;
    a[13] = max;
    a[14] = a1;
    a[15] = a2;

    a[16] = max * max;
    a[17] = p_1;
    a[18] = a1 * a1;
    a[19] = inE4 * p_1;
    a[20] = max * max;
    a[21] = p_1;
    a[22] = a1 * a1;
    a[23] = inE4 * p_1;

    // Use arbitrary values (PoseidonGoldilocksConstants removed in Poseidon2 refactor)
    for (int k = 0; k < 144; ++k)
        Mat[k] = Goldilocks::fromU64(k + 1);

    // product
    for (int l = 0; l < 3; ++l)
    {
        for (int i = 0; i < 4; ++i)
        {

            Goldilocks::Element sum = Goldilocks::zero();
            for (int k = 0; k < 3; ++k)
            {
                for (int j = 0; j < 4; ++j)
                {
                    sum = sum + (Mat[(l * 4 + i) * 12 + k * 4 + j] * a[k * 8 + j]);
                }
            }
            b[l * 8 + i] = sum;
            b[l * 8 + 4 + i] = sum;
        }
    }

    // avx product
    __m512i a0_;
    __m512i a1_;
    __m512i a2_;

    Goldilocks::load_avx512(a0_, &(a[0]));
    Goldilocks::load_avx512(a1_, &(a[8]));
    Goldilocks::load_avx512(a2_, &(a[16]));
    Goldilocks::mmult_avx512(a0_, a1_, a2_, &(Mat[0]));
    Goldilocks::store_avx512(&(a[0]), a0_);
    Goldilocks::store_avx512(&(a[8]), a1_);
    Goldilocks::store_avx512(&(a[16]), a2_);

    ASSERT_EQ(Goldilocks::toU64(b[0]), Goldilocks::toU64(a[0]));
    ASSERT_EQ(Goldilocks::toU64(b[1]), Goldilocks::toU64(a[1]));
    ASSERT_EQ(Goldilocks::toU64(b[2]), Goldilocks::toU64(a[2]));
    ASSERT_EQ(Goldilocks::toU64(b[3]), Goldilocks::toU64(a[3]));
    ASSERT_EQ(Goldilocks::toU64(b[4]), Goldilocks::toU64(a[4]));
    ASSERT_EQ(Goldilocks::toU64(b[5]), Goldilocks::toU64(a[5]));
    ASSERT_EQ(Goldilocks::toU64(b[6]), Goldilocks::toU64(a[6]));
    ASSERT_EQ(Goldilocks::toU64(b[7]), Goldilocks::toU64(a[7]));
    ASSERT_EQ(Goldilocks::toU64(b[8]), Goldilocks::toU64(a[8]));
    ASSERT_EQ(Goldilocks::toU64(b[9]), Goldilocks::toU64(a[9]));
    ASSERT_EQ(Goldilocks::toU64(b[10]), Goldilocks::toU64(a[10]));
    ASSERT_EQ(Goldilocks::toU64(b[11]), Goldilocks::toU64(a[11]));
    ASSERT_EQ(Goldilocks::toU64(b[12]), Goldilocks::toU64(a[12]));
    ASSERT_EQ(Goldilocks::toU64(b[13]), Goldilocks::toU64(a[13]));
    ASSERT_EQ(Goldilocks::toU64(b[14]), Goldilocks::toU64(a[14]));
    ASSERT_EQ(Goldilocks::toU64(b[15]), Goldilocks::toU64(a[15]));
    ASSERT_EQ(Goldilocks::toU64(b[16]), Goldilocks::toU64(a[16]));
    ASSERT_EQ(Goldilocks::toU64(b[17]), Goldilocks::toU64(a[17]));
    ASSERT_EQ(Goldilocks::toU64(b[18]), Goldilocks::toU64(a[18]));
    ASSERT_EQ(Goldilocks::toU64(b[19]), Goldilocks::toU64(a[19]));
    ASSERT_EQ(Goldilocks::toU64(b[20]), Goldilocks::toU64(a[20]));
    ASSERT_EQ(Goldilocks::toU64(b[21]), Goldilocks::toU64(a[21]));
    ASSERT_EQ(Goldilocks::toU64(b[22]), Goldilocks::toU64(a[22]));
    ASSERT_EQ(Goldilocks::toU64(b[23]), Goldilocks::toU64(a[23]));

    // avx product coefs small
    a[0] = inE1;
    a[1] = inE2;
    a[2] = Goldilocks::one();
    a[3] = inE3;
    a[4] = inE1;
    a[5] = inE2;
    a[6] = Goldilocks::one();
    a[7] = inE3;

    a[8] = inE4;
    a[9] = max;
    a[10] = a1;
    a[11] = a2;
    a[12] = inE4;
    a[13] = max;
    a[14] = a1;
    a[15] = a2;

    a[16] = max * max;
    a[17] = p_1;
    a[18] = a1 * a1;
    a[19] = inE4 * p_1;
    a[20] = max * max;
    a[21] = p_1;
    a[22] = a1 * a1;
    a[23] = inE4 * p_1;

    Goldilocks::load_avx512(a0_, &(a[0]));
    Goldilocks::load_avx512(a1_, &(a[8]));
    Goldilocks::load_avx512(a2_, &(a[16]));
    Goldilocks::mmult_avx512_8(a0_, a1_, a2_, &(Mat[0]));
    Goldilocks::store_avx512(&(a[0]), a0_);
    Goldilocks::store_avx512(&(a[8]), a1_);
    Goldilocks::store_avx512(&(a[16]), a2_);

    ASSERT_EQ(Goldilocks::toU64(b[0]), Goldilocks::toU64(a[0]));
    ASSERT_EQ(Goldilocks::toU64(b[1]), Goldilocks::toU64(a[1]));
    ASSERT_EQ(Goldilocks::toU64(b[2]), Goldilocks::toU64(a[2]));
    ASSERT_EQ(Goldilocks::toU64(b[3]), Goldilocks::toU64(a[3]));
    ASSERT_EQ(Goldilocks::toU64(b[4]), Goldilocks::toU64(a[4]));
    ASSERT_EQ(Goldilocks::toU64(b[5]), Goldilocks::toU64(a[5]));
    ASSERT_EQ(Goldilocks::toU64(b[6]), Goldilocks::toU64(a[6]));
    ASSERT_EQ(Goldilocks::toU64(b[7]), Goldilocks::toU64(a[7]));
    ASSERT_EQ(Goldilocks::toU64(b[8]), Goldilocks::toU64(a[8]));
    ASSERT_EQ(Goldilocks::toU64(b[9]), Goldilocks::toU64(a[9]));
    ASSERT_EQ(Goldilocks::toU64(b[10]), Goldilocks::toU64(a[10]));
    ASSERT_EQ(Goldilocks::toU64(b[11]), Goldilocks::toU64(a[11]));
    ASSERT_EQ(Goldilocks::toU64(b[12]), Goldilocks::toU64(a[12]));
    ASSERT_EQ(Goldilocks::toU64(b[13]), Goldilocks::toU64(a[13]));
    ASSERT_EQ(Goldilocks::toU64(b[14]), Goldilocks::toU64(a[14]));
    ASSERT_EQ(Goldilocks::toU64(b[15]), Goldilocks::toU64(a[15]));
    ASSERT_EQ(Goldilocks::toU64(b[16]), Goldilocks::toU64(a[16]));
    ASSERT_EQ(Goldilocks::toU64(b[17]), Goldilocks::toU64(a[17]));
    ASSERT_EQ(Goldilocks::toU64(b[18]), Goldilocks::toU64(a[18]));
    ASSERT_EQ(Goldilocks::toU64(b[19]), Goldilocks::toU64(a[19]));
    ASSERT_EQ(Goldilocks::toU64(b[20]), Goldilocks::toU64(a[20]));
    ASSERT_EQ(Goldilocks::toU64(b[21]), Goldilocks::toU64(a[21]));
    ASSERT_EQ(Goldilocks::toU64(b[22]), Goldilocks::toU64(a[22]));
    ASSERT_EQ(Goldilocks::toU64(b[23]), Goldilocks::toU64(a[23]));

    free(a);
    free(Mat);
    free(b);
}
#endif

TEST(GOLDILOCKS_TEST, div)
{
    uint64_t in1 = 10;
    int32_t in2 = 5;
    std::string in3 = "92233720347072921606"; // GOLDILOCKS_PRIME * 5 + 1
    int32_t in4 = -12;

    Goldilocks::Element inE1 = Goldilocks::fromU64(in1);
    Goldilocks::Element inE2 = Goldilocks::fromS32(in2);
    Goldilocks::Element inE3 = Goldilocks::fromString(in3);
    Goldilocks::Element inE4 = Goldilocks::fromS32(in4);
    Goldilocks::Element inE5 = Goldilocks::fromS32(in4);
    Goldilocks::Element inE6 = Goldilocks::fromS32(in4);

    ASSERT_EQ(Goldilocks::toU64(inE1 / inE2), in1 / in2);
    ASSERT_EQ(Goldilocks::toU64(inE1 / inE2 / inE3), in1 / in2); // 10 / 2 / ( 0 + 1 ) = 10 / 2
    ASSERT_EQ(Goldilocks::toU64(inE1 / inE2 / inE3 / inE4), 0X2AAAAAAA80000000);
    ASSERT_EQ(Goldilocks::toU64(inE5 / inE6), 1);
    ASSERT_EQ(Goldilocks::toU64(Goldilocks::one() / inE6), 0X1555555540000000);
}
TEST(GOLDILOCKS_TEST, inv)
{
    uint64_t in1 = 5;
    std::string in2 = "18446744069414584326"; // 0xFFFFFFFF00000001n + 5n

    Goldilocks::Element input1 = Goldilocks::one();
    Goldilocks::Element inv1 = Goldilocks::inv(input1);
    Goldilocks::Element res1 = input1 * inv1;

    Goldilocks::Element input5 = Goldilocks::fromU64(in1);
    Goldilocks::Element inv5 = Goldilocks::inv(input5);
    Goldilocks::Element res5 = input5 * inv5;

    ASSERT_EQ(res1, Goldilocks::one());
    ASSERT_EQ(res5, Goldilocks::one());

    Goldilocks::Element inE1 = Goldilocks::fromString(std::to_string(in1));
    Goldilocks::Element inE1_plus_p = Goldilocks::fromString(in2);

    ASSERT_EQ(Goldilocks::inv(inE1_plus_p) * inE1, Goldilocks::one());
    ASSERT_EQ(Goldilocks::inv(inE1), Goldilocks::inv(inE1_plus_p));
}

