#pragma once
// The aggregation const cache's bookkeeping, kept free of CUDA so it can be unit-tested: slot
// layout (slot 0 large, then recursive-sized LRU slots, then pinned slots), tags and the pick rule.
// Keys are opaque set identities chosen by the caller (non-negative; -1 means free).
#include <cstdint>

struct ConstSlotTable {
    static constexpr uint32_t MAX_SLOTS = 64;
    uint32_t nSlots = 0;
    uint32_t nLru = 0;
    uint64_t base = 0;
    uint64_t offset[MAX_SLOTS] = {};
    uint64_t elems[MAX_SLOTS] = {};
    int64_t pinnedKey[MAX_SLOTS];
    int64_t tag[MAX_SLOTS];
    uint64_t held[MAX_SLOTS] = {};  // elements of the set a tagged slot holds
    uint64_t lastUse[MAX_SLOTS];
    uint32_t pending[MAX_SLOTS] = {};
    uint64_t useClock = 0;

    ConstSlotTable() { layout(0, 0, 0, 0); }

    uint64_t end() const { return nSlots == 0 ? 0 : offset[nSlots - 1] + elems[nSlots - 1]; }

    void layout(uint64_t base_, uint64_t bigElems, uint64_t normalElems, uint32_t nLru_) {
        base = base_;
        nLru = nLru_;
        nSlots = nLru_;
        uint64_t off = 0;
        for (uint32_t s = 0; s < nLru_; s++) {
            offset[s] = off;
            elems[s] = s == 0 ? bigElems : normalElems;
            off += elems[s];
        }
        for (uint32_t s = 0; s < MAX_SLOTS; s++) pinnedKey[s] = -1;
        clearTags();
    }

    int32_t pin(int64_t key, uint64_t elemsPinned) {
        if (nSlots >= MAX_SLOTS) return -1;
        const uint32_t s = nSlots;
        offset[s] = end();
        nSlots++;
        elems[s] = elemsPinned;
        pinnedKey[s] = key;
        tag[s] = -1; lastUse[s] = 0; pending[s] = 0;
        return (int32_t)s;
    }

    int32_t find(int64_t key) const {
        for (uint32_t s = 0; s < nSlots; s++) if (tag[s] == key) return (int32_t)s;
        return -1;
    }

    int32_t pinnedSlot(int64_t key) const {
        for (uint32_t s = nLru; s < nSlots; s++) if (pinnedKey[s] == key) return (int32_t)s;
        return -1;
    }

    void fill(int32_t s, int64_t key, uint64_t elemsHeld) { tag[s] = key; held[s] = elemsHeld; }

    // Least recently used free LRU slot that fits, normal slots winning ties. Slot 0 is passed over
    // while it holds a set no normal slot fits, the sets it alone can take.
    int32_t pickLru(uint64_t needElems) const {
        int32_t best = -1;
        uint64_t bestAge = UINT64_MAX;
        for (uint32_t s = 1; s < nLru; s++) {
            if (pending[s] != 0 || elems[s] < needElems) continue;
            const uint64_t age = tag[s] == -1 ? 0 : lastUse[s];
            if (age < bestAge) { bestAge = age; best = (int32_t)s; }
        }
        if (nLru == 0 || pending[0] != 0 || elems[0] < needElems) return best;
        const bool holdsLarge = tag[0] != -1 && nLru > 1 && held[0] > elems[1];
        const uint64_t age0 = tag[0] == -1 ? 0 : lastUse[0];
        return (best < 0 || (!holdsLarge && age0 < bestAge)) ? 0 : best;
    }

    // Slot 0 is the largest LRU slot (configure enforces it).
    bool fitsAnyLru(uint64_t needElems) const { return nLru > 0 && elems[0] >= needElems; }

    void clearTags() { for (uint32_t s = 0; s < MAX_SLOTS; s++) { tag[s] = -1; held[s] = 0; lastUse[s] = 0; pending[s] = 0; } }

    void clearBelow(uint64_t toElem) {
        for (uint32_t s = 0; s < nSlots; s++)
            if (base + offset[s] < toElem) { tag[s] = -1; lastUse[s] = 0; }
    }
};
