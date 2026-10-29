#pragma once
// Which fixed columns a stream's aux trace holds, kept free of CUDA so it can be unit-tested.
#include <cstdint>
#include <string>

struct FixedSlotState {
    // Which fixed columns this stream's aux trace holds, not which air it last served: airs
    // with identical fixed share a const-pols slot, so the offset is the key. constAggBuffer
    // separates the two const buffers (unrelated offsets); constRecurserId holds a const-cached
    // set's "cache:<key>" (its offset is UINT64_MAX). custom_fixed is per-air and stays keyed on the air.
    uint64_t constPolsOffset = UINT64_MAX; // UINT64_MAX = nothing cached
    // Where the unpacked pols land in the aux trace. Part of the key because two airs can in
    // principle share a slot yet lay out ("const", false) differently.
    uint64_t constAuxOffset = 0;
    bool constAggBuffer = false;
    std::string constRecurserId;
    // ("const", true) holds this slot's tree. Separate from the pols because the
    // constraint-verification path unpacks the pols without loading the tree.
    bool constTreeResident = false;

    // No fixed slot (zero-fixed air): its proof overwrites the aux trace, so no later adoptFixedSlot may match.
    void dropFixedSlot() {
        constPolsOffset = UINT64_MAX;
        constAuxOffset = UINT64_MAX;
        constRecurserId = "";
        constTreeResident = false;
    }

    // Claim this slot; true if it was already claimed, i.e. the unpacked const pols still
    // apply -- even to a different air sharing them. A change also drops constTreeResident.
    bool adoptFixedSlot(uint64_t offset, uint64_t auxOffset, bool aggBuffer, const std::string &recurser) {
        if (constPolsOffset == offset && constAuxOffset == auxOffset && constAggBuffer == aggBuffer
            && constRecurserId == recurser) return true;
        constPolsOffset = offset;
        constAuxOffset = auxOffset;
        constAggBuffer = aggBuffer;
        constRecurserId = recurser;
        constTreeResident = false;
        return false;
    }

    // A basic proof's claim: true if the unpacked const pols it needs are still in the aux trace.
    bool claimBasicFixedSlot(bool hasConstants, uint64_t offset, uint64_t auxOffset) {
        if (!hasConstants) {
            dropFixedSlot();
            return false;
        }
        return adoptFixedSlot(offset, auxOffset, false, "");
    }
};

// Whether an air's packed const pols are on the device; a const-cached set (UINT64_MAX) is not.
inline bool constPolsResident(uint64_t constPolsOffset, bool hasConstants) {
    return !hasConstants || constPolsOffset != UINT64_MAX;
}
