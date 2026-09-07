#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "unpack_indexed_row.hpp"

// Coverage for the row-level indexed unpack shared by the CPU witness path. A row holds
// one instruction index per lane plus its runtime columns, and every instruction-derived
// column is read from the entry its lane's index selects.

namespace {

// Bit-pack `n` values of the given widths into `words` u64s, LSB-first, exactly as the
// generated packed rows are laid out.
void packBits(const std::vector<uint64_t> &vals, const std::vector<uint64_t> &widths, uint64_t words,
              uint64_t *dst)
{
    for (uint64_t i = 0; i < words; i++) dst[i] = 0;
    uint64_t bit = 0;
    for (size_t i = 0; i < vals.size(); i++) {
        const uint64_t w = widths[i];
        const uint64_t v = (w == 64) ? vals[i] : (vals[i] & ((1ull << w) - 1ull));
        const uint64_t widx = bit / 64, off = bit % 64;
        dst[widx] |= v << off;
        if (off + w > 64) dst[widx + 1] |= v >> (64 - off);
        bit += w;
    }
}

uint64_t wordsFor(const std::vector<uint64_t> &widths)
{
    uint64_t bits = 0;
    for (uint64_t w : widths) bits += w;
    return (bits + 63) / 64;
}

// A lane-packed row: fields in declaration order, each expanded lane-major, so a field
// with `sub` sub-columns per lane gives each lane `sub` consecutive columns. Instruction
// fields are lane-stripped in the entry.
struct Layout {
    std::vector<uint64_t> widths;   // per output column
    std::vector<uint8_t> colSource; // 0 = row stream, 1 = table
    std::vector<uint8_t> colLane;   // which lane's index selects the entry
    std::vector<uint64_t> rowWidths, entryWidths;
    uint64_t lanes, indexBits = 32;

    // field = (width, sub-columns per lane, instruction-derived)
    Layout(uint64_t lanes_, const std::vector<std::tuple<uint64_t, uint64_t, bool>> &fields) : lanes(lanes_)
    {
        for (uint64_t l = 0; l < lanes; l++) rowWidths.push_back(indexBits);
        for (auto &[width, sub, instr] : fields) {
            for (uint64_t l = 0; l < lanes; l++) {
                for (uint64_t s = 0; s < sub; s++) {
                    widths.push_back(width);
                    colSource.push_back(instr ? 1 : 0);
                    colLane.push_back(static_cast<uint8_t>(l));
                    if (!instr) rowWidths.push_back(width);
                }
            }
            // One entry holds a single lane's worth of the field.
            if (instr)
                for (uint64_t s = 0; s < sub; s++) entryWidths.push_back(width);
        }
    }
};

} // namespace

// Single lane is the pre-lane shape: one index, one entry, interleaved sources.
TEST(UNPACK_INDEXED_CPU, single_lane_reads_the_only_entry)
{
    Layout L(1, {{16, 1, false}, {32, 1, true}, {1, 1, false}, {64, 1, true}});
    const uint64_t rowWords = wordsFor(L.rowWidths), entWords = wordsFor(L.entryWidths);

    std::vector<uint64_t> entry{0xCAFEBABEull, 0x0123456789ABCDEFull};
    std::vector<uint64_t> table(2 * entWords, 0);
    packBits(entry, L.entryWidths, entWords, &table[1 * entWords]);

    std::vector<uint64_t> row{1 /*index*/, 0xBEEF, 1};
    std::vector<uint64_t> packed(rowWords);
    packBits(row, L.rowWidths, rowWords, packed.data());

    std::vector<uint64_t> out(L.widths.size(), 0);
    ASSERT_TRUE(unpackIndexedRow(packed.data(), rowWords, table.data(), entWords, 2, L.indexBits,
                                 L.lanes, L.widths.data(), L.colSource.data(), L.colLane.data(),
                                 L.widths.size(), out.data()));
    EXPECT_EQ(out[0], 0xBEEFull);
    EXPECT_EQ(out[1], 0xCAFEBABEull);
    EXPECT_EQ(out[2], 1ull);
    EXPECT_EQ(out[3], 0x0123456789ABCDEFull);
}

// The point of the lane generalization: one row mixes columns from several entries.
TEST(UNPACK_INDEXED_CPU, each_lane_reads_the_entry_its_own_index_selects)
{
    Layout L(3, {{16, 2, false}, {32, 1, true}, {12, 2, true}, {1, 1, false}, {64, 1, true}});
    const uint64_t rowWords = wordsFor(L.rowWidths), entWords = wordsFor(L.entryWidths);
    const uint64_t nEntries = 4;

    // Entry e: p = 0x1000+e, q = {0x20+e, 0x30+e}, r = 0xAAAA0000+e.
    std::vector<uint64_t> table(nEntries * entWords);
    for (uint64_t e = 0; e < nEntries; e++) {
        std::vector<uint64_t> vals{0x1000 + e, 0x20 + e, 0x30 + e, 0xAAAA0000ull + e};
        packBits(vals, L.entryWidths, entWords, &table[e * entWords]);
    }

    // Lane 0 -> entry 2, lane 1 -> entry 0, lane 2 -> entry 3.
    const uint64_t idx[3] = {2, 0, 3};
    std::vector<uint64_t> row{idx[0], idx[1], idx[2],
                              // rt_a: 2 sub-columns per lane
                              0xA00, 0xA01, 0xA10, 0xA11, 0xA20, 0xA21,
                              // rt_b: 1 per lane
                              1, 0, 1};
    std::vector<uint64_t> packed(rowWords);
    packBits(row, L.rowWidths, rowWords, packed.data());

    std::vector<uint64_t> out(L.widths.size(), 0);
    ASSERT_TRUE(unpackIndexedRow(packed.data(), rowWords, table.data(), entWords, nEntries,
                                 L.indexBits, L.lanes, L.widths.data(), L.colSource.data(),
                                 L.colLane.data(), L.widths.size(), out.data()));

    // Runtime columns, lane-major: rt_a at 0..5, rt_b at 15..17.
    const uint64_t rtA[6] = {0xA00, 0xA01, 0xA10, 0xA11, 0xA20, 0xA21};
    for (uint64_t i = 0; i < 6; i++) EXPECT_EQ(out[i], rtA[i]) << "rt_a column " << i;
    EXPECT_EQ(out[15], 1ull);
    EXPECT_EQ(out[16], 0ull);
    EXPECT_EQ(out[17], 1ull);

    // ins_p: one column per lane, columns 6..8.
    for (uint64_t l = 0; l < 3; l++) EXPECT_EQ(out[6 + l], 0x1000 + idx[l]) << "ins_p lane " << l;
    // ins_q: two sub-columns per lane (9..14), which must stay in order within the lane.
    for (uint64_t l = 0; l < 3; l++) {
        EXPECT_EQ(out[9 + 2 * l], 0x20 + idx[l]) << "ins_q[0] lane " << l;
        EXPECT_EQ(out[9 + 2 * l + 1], 0x30 + idx[l]) << "ins_q[1] lane " << l;
    }
    // ins_r: 64 bits wide, so its read spans words -- columns 18..20.
    for (uint64_t l = 0; l < 3; l++)
        EXPECT_EQ(out[18 + l], 0xAAAA0000ull + idx[l]) << "ins_r lane " << l;
}

// A witness bug can put a stale index in one lane; the caller must be able to name it.
TEST(UNPACK_INDEXED_CPU, reports_the_lane_whose_index_is_out_of_range)
{
    Layout L(2, {{32, 1, true}, {16, 1, false}});
    const uint64_t rowWords = wordsFor(L.rowWidths), entWords = wordsFor(L.entryWidths);
    const uint64_t nEntries = 2;
    std::vector<uint64_t> table(nEntries * entWords, 0);

    std::vector<uint64_t> row{0 /*lane 0 ok*/, 7 /*lane 1 past the table*/, 0xBEEF};
    std::vector<uint64_t> packed(rowWords);
    packBits(row, L.rowWidths, rowWords, packed.data());

    std::vector<uint64_t> out(L.widths.size(), 0);
    uint64_t badLane = 0, badIndex = 0;
    EXPECT_FALSE(unpackIndexedRow(packed.data(), rowWords, table.data(), entWords, nEntries,
                                  L.indexBits, L.lanes, L.widths.data(), L.colSource.data(),
                                  L.colLane.data(), L.widths.size(), out.data(), &badLane,
                                  &badIndex));
    EXPECT_EQ(badLane, 1ull);
    EXPECT_EQ(badIndex, 7ull);
}
