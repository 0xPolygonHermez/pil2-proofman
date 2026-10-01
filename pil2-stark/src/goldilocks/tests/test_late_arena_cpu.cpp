#include <gtest/gtest.h>
#include <cstdint>
#include "late_arena.hpp"

alignas(256) static uint8_t mem[1 << 20];

TEST(LateArena, same_tag_same_address_across_invalidate) {
    lateArenaBind(7, mem, sizeof(mem));
    void *a = nullptr, *b = nullptr;
    ASSERT_EQ(lateArenaAlloc(7, "mul.acc", 1000, &a), LateRc::Ok);
    const uint32_t g = lateArenaGen(7);
    lateArenaInvalidate(7);
    EXPECT_NE(lateArenaGen(7), g);
    ASSERT_EQ(lateArenaAlloc(7, "mul.acc", 1000, &b), LateRc::Ok);
    EXPECT_EQ(a, b);
    lateArenaUnbind(7);
}

TEST(LateArena, allocations_are_aligned_and_disjoint) {
    lateArenaBind(7, mem, sizeof(mem));
    void *a = nullptr, *b = nullptr;
    ASSERT_EQ(lateArenaAlloc(7, "x", 1, &a), LateRc::Ok);
    ASSERT_EQ(lateArenaAlloc(7, "y", 1, &b), LateRc::Ok);
    EXPECT_EQ((uintptr_t)a % LATE_ALIGN, 0u);
    EXPECT_EQ((uint8_t *)b - (uint8_t *)a, (ptrdiff_t)LATE_ALIGN);
    lateArenaUnbind(7);
}

TEST(LateArena, overrun_is_refused) {
    lateArenaBind(7, mem, 512);
    void *a = nullptr;
    ASSERT_EQ(lateArenaAlloc(7, "x", 256, &a), LateRc::Ok);
    EXPECT_EQ(lateArenaAlloc(7, "y", 512, &a), LateRc::Overrun);
    lateArenaUnbind(7);
}

TEST(LateArena, tag_regrow_beyond_reservation_is_refused) {
    lateArenaBind(7, mem, sizeof(mem));
    void *a = nullptr;
    ASSERT_EQ(lateArenaAlloc(7, "mul.map.7", 1000, &a), LateRc::Ok);
    EXPECT_EQ(lateArenaAlloc(7, "mul.map.7", 999, &a), LateRc::Ok);      // smaller fits
    EXPECT_EQ(lateArenaAlloc(7, "mul.map.7", 5000, &a), LateRc::Regrow);
    lateArenaUnbind(7);
}

TEST(LateArena, unbound_arena_reports_unbound) {
    void *a = nullptr;
    EXPECT_EQ(lateArenaAlloc(99, "x", 8, &a), LateRc::Unbound);
}

TEST(LateArena, rebind_forgets_tags) {
    lateArenaBind(7, mem, sizeof(mem));
    void *a = nullptr, *b = nullptr;
    lateArenaAlloc(7, "x", 4096, &a);
    lateArenaUnbind(7);
    lateArenaBind(7, mem + 512, sizeof(mem) - 512);
    lateArenaAlloc(7, "x", 4096, &b);
    EXPECT_EQ((uint8_t *)b, mem + 512);
    lateArenaUnbind(7);
}

// The late region starts right after the basic fixed pols, so its base is only 8-byte aligned.
TEST(LateArena, misaligned_base_aligns_absolute_address) {
    uint8_t *base = mem + 8;
    lateArenaBind(7, base, 1024);
    void *a = nullptr, *b = nullptr, *c = nullptr;
    ASSERT_EQ(lateArenaAlloc(7, "x", 1, &a), LateRc::Ok);
    ASSERT_EQ(lateArenaAlloc(7, "y", 300, &b), LateRc::Ok);
    EXPECT_EQ((uintptr_t)a % LATE_ALIGN, 0u);
    EXPECT_EQ((uintptr_t)b % LATE_ALIGN, 0u);
    EXPECT_EQ((uint8_t *)a, mem + 256);
    EXPECT_GE((uint8_t *)b, (uint8_t *)a + 1);
    // 1024 B from mem+8 ends at mem+1032; "y" ends at mem+1024, so 256 more no longer fit.
    EXPECT_EQ(lateArenaAlloc(7, "z", 8, &c), LateRc::Overrun);
    EXPECT_LE(lateArenaUsed(7), 1024u);
    lateArenaUnbind(7);
}

TEST(LateArena, empty_region_stays_unbound) {
    lateArenaBind(7, mem, sizeof(mem));
    const uint32_t g = lateArenaGen(7);
    lateArenaBind(7, mem, 0);
    EXPECT_NE(lateArenaGen(7), g);
    void *a = nullptr;
    EXPECT_EQ(lateArenaAlloc(7, "x", 8, &a), LateRc::Unbound);
    EXPECT_EQ(lateArenaCap(7), 0u);
    lateArenaUnbind(7);
}

TEST(LateArena, owns_only_its_bound_range) {
    static uint64_t other;
    EXPECT_FALSE(lateArenaBound(7));
    EXPECT_FALSE(lateArenaOwns(7, mem));
    lateArenaBind(7, mem, 4096);
    EXPECT_TRUE(lateArenaBound(7));
    void *a = nullptr;
    ASSERT_EQ(lateArenaAlloc(7, "x", 8, &a), LateRc::Ok);
    EXPECT_TRUE(lateArenaOwns(7, a));
    EXPECT_FALSE(lateArenaOwns(7, mem + 4096));
    EXPECT_FALSE(lateArenaOwns(7, &other));
    lateArenaUnbind(7);
    EXPECT_FALSE(lateArenaOwns(7, a));
}

TEST(LateArenaDeathTest, rebind_with_tags_is_refused) {
    lateArenaBind(8, mem, sizeof(mem));
    void *a = nullptr;
    ASSERT_EQ(lateArenaAlloc(8, "x", 8, &a), LateRc::Ok);
    EXPECT_DEATH(lateArenaBind(8, mem, sizeof(mem)), "still bound");
    lateArenaUnbind(8);
}

// ZisK-like mul mirrors: acc, three maps, oob, and (multi-GPU) the peer staging.
static const std::vector<uint64_t> ZISK_MUL{300 << 20, 167 << 20, 50 << 20, 41 << 20, 128};

TEST(LateArena, plan_single_gpu_has_no_peer_staging) {
    EXPECT_EQ(latePlanBytes(ZISK_MUL), (300u << 20) + (167u << 20) + (50u << 20) + (41u << 20) + 256 + 256);
}

TEST(LateArena, plan_multi_gpu_adds_peer_staging) {
    std::vector<uint64_t> peers = ZISK_MUL;
    peers.push_back(64 << 20);
    EXPECT_EQ(latePlanBytes(peers) - latePlanBytes(ZISK_MUL), 64u << 20);
}

TEST(LateArena, plan_blocks_are_aligned) {
    EXPECT_EQ(latePlanBytes({3 * 8, 0, 1 * 8}), 256u + 0 + 256 + 256);
}
