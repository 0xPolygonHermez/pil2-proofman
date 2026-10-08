#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "pack_columns.hpp"

TEST(PackColumns, WidthsRowMajor)
{
    // 3 rows x 3 cols, row-major: col0 = {0,0,0}, col1 = {1,2,3}, col2 = {0,1<<40,5}
    std::vector<uint64_t> rm = {0, 1, 0,
                                0, 2, 1ULL << 40,
                                0, 3, 5};
    std::vector<uint64_t> widths(3, 0);
    uint64_t wpr = packWidthsRowMajor(rm.data(), 3, 3, widths.data());
    EXPECT_EQ(widths[0], 1ULL);   // an all-zero column still costs one bit
    EXPECT_EQ(widths[1], 2ULL);   // max 3 -> 2 bits
    EXPECT_EQ(widths[2], 41ULL);  // 1<<40 -> 41 bits
    EXPECT_EQ(wpr, 1ULL);         // 44 bits -> one word
}

TEST(PackColumns, RoundTrip)
{
    const uint64_t nRows = 5000;
    const uint64_t nCols = 12;  // the ZisK ROM width
    // Widths chosen to straddle word boundaries the way the real ROM does (465 bits -> 8 words).
    const uint64_t bits[nCols] = {1, 32, 32, 32, 64, 32, 32, 32, 64, 64, 64, 16};
    std::mt19937_64 rng(12345);

    std::vector<uint64_t> rm(nRows * nCols);
    for (uint64_t r = 0; r < nRows; ++r)
        for (uint64_t c = 0; c < nCols; ++c)
            rm[r * nCols + c] = (bits[c] == 64) ? rng() : (rng() & ((1ULL << bits[c]) - 1ULL));
    // Force each column to actually reach its nominal width.
    for (uint64_t c = 0; c < nCols; ++c)
        rm[c] = (bits[c] == 64) ? ~0ULL : ((1ULL << bits[c]) - 1ULL);

    std::vector<uint64_t> widths(nCols, 0);
    uint64_t wpr = packWidthsRowMajor(rm.data(), nRows, nCols, widths.data());
    EXPECT_EQ(wpr, 8ULL);
    for (uint64_t c = 0; c < nCols; ++c) EXPECT_EQ(widths[c], bits[c]) << "col " << c;

    std::vector<uint64_t> packed(nRows * wpr, 0);
    packRowsBits(rm.data(), packed.data(), nRows, nCols, widths.data(), wpr);

    std::vector<uint64_t> out(nRows * nCols, 0);
    unpackRowsBits(packed.data(), out.data(), nRows, nCols, widths.data(), wpr);
    EXPECT_EQ(out, rm);
}

TEST(PackColumns, RoundTripSingleWordAndZeroMatrix)
{
    const uint64_t nRows = 7, nCols = 3;
    std::vector<uint64_t> rm(nRows * nCols, 0);
    std::vector<uint64_t> widths(nCols, 0);
    uint64_t wpr = packWidthsRowMajor(rm.data(), nRows, nCols, widths.data());
    EXPECT_EQ(wpr, 1ULL);  // three all-zero columns -> 3 bits -> one word

    std::vector<uint64_t> packed(nRows * wpr, 0), out(nRows * nCols, 1);
    packRowsBits(rm.data(), packed.data(), nRows, nCols, widths.data(), wpr);
    unpackRowsBits(packed.data(), out.data(), nRows, nCols, widths.data(), wpr);
    EXPECT_EQ(out, rm);
}
