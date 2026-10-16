#include <gtest/gtest.h>
#include "const_slot_table.hpp"

static ConstSlotTable table(uint32_t nLru) {
    ConstSlotTable t;
    t.layout(1000, 300, 100, nLru);
    return t;
}

TEST(ConstSlotTable, layout_puts_the_large_slot_first) {
    ConstSlotTable t = table(3);
    EXPECT_EQ(t.nSlots, 3u);
    EXPECT_EQ(t.elems[0], 300u);
    EXPECT_EQ(t.offset[1], 300u);
    EXPECT_EQ(t.offset[2], 400u);
    EXPECT_EQ(t.end(), 500u);
}

TEST(ConstSlotTable, pins_append_above_the_lru) {
    ConstSlotTable t = table(2);
    const int64_t k = 3003;
    EXPECT_EQ(t.pin(k, 120), 2);
    EXPECT_EQ(t.offset[2], 400u);
    EXPECT_EQ(t.end(), 520u);
    EXPECT_EQ(t.pinnedSlot(k), 2);
    EXPECT_EQ(t.nLru, 2u);
}

TEST(ConstSlotTable, small_sets_prefer_normal_slots) {
    ConstSlotTable t = table(3);
    EXPECT_EQ(t.pickLru(100), 1);  // free normal slot before the large one
    t.fill(0, 6, 300); t.lastUse[0] = ++t.useClock;  // a set only slot 0 fits
    t.fill(1, 7, 100); t.lastUse[1] = ++t.useClock;
    t.fill(2, 8, 100); t.lastUse[2] = ++t.useClock;
    EXPECT_EQ(t.pickLru(100), 1);  // LRU among normal slots: slot 0 keeps its large set
    t.pending[1] = 1; t.pending[2] = 1;
    EXPECT_EQ(t.pickLru(100), 0);  // only then the large one
}

TEST(ConstSlotTable, slot_zero_is_lru_while_it_holds_no_large_set) {
    ConstSlotTable t = table(2);
    t.fill(1, 7, 100); t.lastUse[1] = ++t.useClock;
    EXPECT_EQ(t.pickLru(100), 0);  // an empty slot 0 rather than evicting slot 1
    t.fill(0, 8, 100); t.lastUse[0] = ++t.useClock;
    EXPECT_EQ(t.pickLru(100), 1);  // then plain LRU over both
    t.lastUse[1] = ++t.useClock;
    EXPECT_EQ(t.pickLru(100), 0);
}

TEST(ConstSlotTable, large_sets_only_fit_slot_zero) {
    ConstSlotTable t = table(3);
    EXPECT_EQ(t.pickLru(250), 0);
    t.pending[0] = 1;
    EXPECT_EQ(t.pickLru(250), -1);
    EXPECT_TRUE(t.fitsAnyLru(250));
}

TEST(ConstSlotTable, oversize_has_no_slot) {
    ConstSlotTable t = table(3);
    EXPECT_FALSE(t.fitsAnyLru(301));
    EXPECT_EQ(t.pickLru(301), -1);
}

TEST(ConstSlotTable, pending_slots_are_never_picked) {
    ConstSlotTable t = table(2);
    t.pending[0] = 1; t.pending[1] = 1;
    EXPECT_EQ(t.pickLru(50), -1);
}

TEST(ConstSlotTable, pinned_slots_are_not_lru) {
    ConstSlotTable t = table(1);
    t.pin(3003, 100);
    t.pending[0] = 1;
    EXPECT_EQ(t.pickLru(50), -1);
}

TEST(ConstSlotTable, find_hits_any_slot) {
    ConstSlotTable t = table(2);
    const int64_t k = 3003;
    t.pin(k, 100);
    EXPECT_EQ(t.find(k), -1);
    t.tag[2] = k;
    EXPECT_EQ(t.find(k), 2);
}

TEST(ConstSlotTable, forget_keeps_layout_and_pins) {
    ConstSlotTable t = table(2);
    t.pin(3003, 100);
    for (uint32_t s = 0; s < 3; s++) t.tag[s] = (int64_t)s + 10;
    t.forget();
    for (uint32_t s = 0; s < 3; s++) EXPECT_EQ(t.tag[s], -1);
    EXPECT_EQ(t.pinnedSlot(3003), 2);
    EXPECT_EQ(t.nSlots, 3u);
}
