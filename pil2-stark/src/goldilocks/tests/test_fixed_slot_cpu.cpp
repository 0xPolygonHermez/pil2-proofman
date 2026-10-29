#include <gtest/gtest.h>
#include "fixed_slot.hpp"

namespace {
const uint64_t SLOT_A = 4096, AUX_A = 100;
}

TEST(FixedSlot, same_slot_on_one_stream_reuses_the_unpacked_const_pols) {
    FixedSlotState s;
    EXPECT_FALSE(s.claimBasicFixedSlot(true, SLOT_A, AUX_A));
    s.constTreeResident = true;
    EXPECT_TRUE(s.claimBasicFixedSlot(true, SLOT_A, AUX_A));
    EXPECT_TRUE(s.constTreeResident);
}

TEST(FixedSlot, zero_fixed_proof_between_two_proofs_of_a_slot_forces_a_reload) {
    FixedSlotState s;
    EXPECT_FALSE(s.claimBasicFixedSlot(true, SLOT_A, AUX_A));
    s.constTreeResident = true;
    // the zero-fixed air passes the offset its loader never set
    EXPECT_FALSE(s.claimBasicFixedSlot(false, UINT64_MAX, AUX_A));
    EXPECT_FALSE(s.constTreeResident);
    EXPECT_FALSE(s.claimBasicFixedSlot(true, SLOT_A, AUX_A)) << "A reused const pols a zero-fixed proof overwrote";
    EXPECT_FALSE(s.constTreeResident);
}

TEST(FixedSlot, zero_fixed_proofs_never_match_each_other_or_a_const_cached_set) {
    FixedSlotState s;
    EXPECT_FALSE(s.claimBasicFixedSlot(false, UINT64_MAX, AUX_A));
    EXPECT_FALSE(s.claimBasicFixedSlot(false, UINT64_MAX, AUX_A));
    // a const-cached set has offset UINT64_MAX too, but a real aux offset
    EXPECT_FALSE(s.claimBasicFixedSlot(true, UINT64_MAX, AUX_A));
}

TEST(FixedSlot, const_pols_resident) {
    EXPECT_TRUE(constPolsResident(SLOT_A, true));
    EXPECT_FALSE(constPolsResident(UINT64_MAX, true)) << "a const-cached set is not resident";
    EXPECT_TRUE(constPolsResident(UINT64_MAX, false)) << "a zero-fixed air has nothing to load";
}
