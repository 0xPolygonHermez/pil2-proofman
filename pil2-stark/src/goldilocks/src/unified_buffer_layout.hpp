#ifndef UNIFIED_BUFFER_LAYOUT_HPP
#define UNIFIED_BUFFER_LAYOUT_HPP

#include <cstdint>

// The one place that knows the unified buffer's order.
namespace ubl {

constexpr uint64_t AUX_ALIGN = 1ull << 20;  // slot bases must stay at least 16-byte aligned

struct In {  // bytes
    uint64_t basicConst = 0;
    uint64_t lateRegion = 0;
    uint64_t prefetch = 0;
    uint64_t auxTotal = 0;        // sum of the basic streams
    uint64_t recursiveTotal = 0;  // 0 under phase B
    uint64_t slots = 0;           // nSlots * slotBytes
    uint64_t aggConst = 0;
    uint64_t snarkPad = 0;
};

struct Layout {  // byte offsets from the buffer base; the basic fixed pols sit at 0
    uint64_t late = 0, prefetch = 0;
    uint64_t auxBase = 0, slotsBase = 0, mopsBase = 0;
    uint64_t recursiveBase = 0, aggBase = 0, end = 0;
};

inline uint64_t alignUp(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

inline Layout plan(const In& in) {
    Layout L;
    L.late = in.basicConst;
    L.prefetch = L.late + in.lateRegion;
    L.auxBase = alignUp(L.prefetch + in.prefetch, AUX_ALIGN);
    L.slotsBase = L.auxBase;
    L.mopsBase = L.auxBase + in.slots;
    L.recursiveBase = L.auxBase + in.auxTotal;
    L.aggBase = L.recursiveBase + in.recursiveTotal;
    L.end = L.aggBase + in.aggConst + in.snarkPad;
    return L;
}

inline uint64_t slotOffset(const Layout& L, uint64_t slotBytes, uint64_t j) { return L.slotsBase + j * slotBytes; }

constexpr uint64_t SLOT_SCRATCH_ALIGN = 256;

struct SlotAir {  // byte offsets from the slot base, for one air's commit
    uint64_t sideOff = 0, constOff = 0, customOff = 0, end = 0;
};

// Per air: its scratch follows its own commit area, so a slot holds the largest single air, not the sum of maxima.
inline SlotAir slotAir(uint64_t commitBytes, uint64_t sideBytes, uint64_t constBytes, uint64_t customBytes) {
    SlotAir s;
    s.sideOff = alignUp(commitBytes, SLOT_SCRATCH_ALIGN);
    s.constOff = alignUp(s.sideOff + sideBytes, SLOT_SCRATCH_ALIGN);
    s.customOff = alignUp(s.constOff + constBytes, SLOT_SCRATCH_ALIGN);
    s.end = s.customOff + customBytes;
    return s;
}

// relStart/relEnd are relative to auxBase.
inline bool overlapsSlots(const Layout& L, uint64_t relStart, uint64_t relEnd) {
    const uint64_t slotsEnd = L.mopsBase - L.auxBase;
    return relStart < slotsEnd && relEnd > 0;
}

}  // namespace ubl

#endif
