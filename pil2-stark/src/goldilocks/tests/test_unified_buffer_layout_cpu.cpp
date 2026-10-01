#include <gtest/gtest.h>
#include <algorithm>
#include "unified_buffer_layout.hpp"

static constexpr uint64_t MiB = 1ull << 20, GiB = 1ull << 30;

// The measured ZisK/5090 shape (spec, "Projected layout").
static ubl::In zisk() {
    ubl::In in{};
    in.basicConst = 1257 * MiB;        // 1.171 GiB, deliberately not MiB-aligned below
    in.basicConst += 8 * 3;            // odd element count
    in.lateRegion = 651 * MiB;         // the mul mirrors only; the slot scratch is in the slots
    in.prefetch = 3584 * MiB;
    in.auxTotal = 22892 * MiB;         // one basic stream
    in.recursiveTotal = 0;             // phase B: halves alias the stream
    in.slots = 2 * 2433 * MiB;         // per-air slot, scratch included
    in.aggConst = 2668 * MiB;
    in.snarkPad = 0;
    return in;
}

TEST(UnifiedBufferLayout, regions_are_ordered_and_contiguous) {
    const ubl::In in = zisk();
    const ubl::Layout L = ubl::plan(in);
    EXPECT_EQ(L.late, in.basicConst);
    EXPECT_EQ(L.prefetch, L.late + in.lateRegion);
    EXPECT_EQ(L.auxBase % ubl::AUX_ALIGN, 0u);
    EXPECT_GE(L.auxBase, L.prefetch + in.prefetch);
    EXPECT_LT(L.auxBase - (L.prefetch + in.prefetch), ubl::AUX_ALIGN);
    EXPECT_EQ(L.slotsBase, L.auxBase);
    EXPECT_EQ(L.mopsBase, L.auxBase + in.slots);
    EXPECT_EQ(L.recursiveBase, L.auxBase + in.auxTotal);
    EXPECT_EQ(L.aggBase, L.recursiveBase + in.recursiveTotal);
    EXPECT_EQ(L.end, L.aggBase + in.aggConst + in.snarkPad);
}

TEST(UnifiedBufferLayout, slot_offsets_stay_inside_the_first_stream) {
    const ubl::In in = zisk();
    const ubl::Layout L = ubl::plan(in);
    const uint64_t slotBytes = in.slots / 2;
    EXPECT_EQ(ubl::slotOffset(L, slotBytes, 0), L.auxBase);
    EXPECT_EQ(ubl::slotOffset(L, slotBytes, 1), L.auxBase + slotBytes);
    EXPECT_LE(ubl::slotOffset(L, slotBytes, 1) + slotBytes, L.auxBase + in.auxTotal);
}

TEST(UnifiedBufferLayout, mops_window_is_the_top) {
    const ubl::Layout L = ubl::plan(zisk());
    EXPECT_EQ(L.end - L.mopsBase, L.end - L.auxBase - zisk().slots);
    EXPECT_GE(L.end - L.auxBase, 22 * GiB);  // MOPS_FLOOR_BYTES semantics hold on this shape
}

TEST(UnifiedBufferLayout, overlap_is_relative_to_aux_base) {
    const ubl::Layout L = ubl::plan(zisk());
    EXPECT_TRUE(ubl::overlapsSlots(L, 0, 1));                  // first stream starts on the slots
    EXPECT_FALSE(ubl::overlapsSlots(L, zisk().slots, zisk().slots + MiB));
}

TEST(UnifiedBufferLayout, multi_stream_recursive_layout_is_disjoint) {
    ubl::In in = zisk();
    in.auxTotal = 3 * 6 * GiB;          // three basic streams
    in.recursiveTotal = 2 * 7 * GiB;    // two dedicated recursive streams (no phase B)
    in.snarkPad = 64 * MiB;
    const ubl::Layout L = ubl::plan(in);
    EXPECT_EQ(L.recursiveBase, L.auxBase + in.auxTotal);
    EXPECT_EQ(L.aggBase, L.recursiveBase + in.recursiveTotal);
    EXPECT_EQ(L.end, L.aggBase + in.aggConst + in.snarkPad);
}

TEST(UnifiedBufferLayout, no_aggregation_layout) {
    ubl::In in = zisk();
    in.aggConst = 0;
    in.lateRegion = 0;
    in.prefetch = 0;
    const ubl::Layout L = ubl::plan(in);
    EXPECT_EQ(L.aggBase, L.end);
    EXPECT_EQ(L.late, L.prefetch);
}

TEST(UnifiedBufferLayout, element_offsets_round_trip) {
    // The allocator stores element pointers; a byte offset that is not a multiple of 8 would truncate.
    const ubl::Layout L = ubl::plan(zisk());
    for (uint64_t o : {L.late, L.prefetch, L.auxBase, L.mopsBase, L.aggBase, L.end}) EXPECT_EQ(o % 8, 0u);
}

TEST(UnifiedBufferLayout, slot_scratch_follows_the_air_commit_area_aligned) {
    const ubl::SlotAir s = ubl::slotAir(1000, 3 * 8, 1 * 8, 2 * 8);
    EXPECT_EQ(s.sideOff, 1024u);
    EXPECT_EQ(s.constOff, 1280u);
    EXPECT_EQ(s.customOff, 1536u);
    EXPECT_EQ(s.end, 1552u);
    for (uint64_t o : {s.sideOff, s.constOff, s.customOff}) EXPECT_EQ(o % ubl::SLOT_SCRATCH_ALIGN, 0u);
}

TEST(UnifiedBufferLayout, slot_without_scratch_is_its_commit_area) {
    const ubl::SlotAir s = ubl::slotAir(4096, 0, 0, 0);
    EXPECT_EQ(s.sideOff, 4096u);
    EXPECT_EQ(s.customOff, 4096u);
    EXPECT_EQ(s.end, 4096u);
}

// A wide air without scratch and a narrow one with it: the slot is the larger air, not the sum of maxima.
TEST(UnifiedBufferLayout, per_air_slot_is_below_the_sum_of_maxima) {
    const ubl::SlotAir wide = ubl::slotAir(2288 * MiB, 0, 0, 0);
    const ubl::SlotAir narrow = ubl::slotAir(600 * MiB, 64 * MiB, 256 * MiB, 64 * MiB);
    const uint64_t perAir = std::max(wide.end, narrow.end);
    const uint64_t maxima = ubl::slotAir(2288 * MiB, 64 * MiB, 256 * MiB, 64 * MiB).end;
    EXPECT_EQ(perAir, 2288 * MiB);
    EXPECT_EQ(maxima - perAir, 384 * MiB);
}
