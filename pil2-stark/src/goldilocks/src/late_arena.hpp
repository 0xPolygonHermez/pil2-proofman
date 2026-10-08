#ifndef LATE_ARENA_HPP
#define LATE_ARENA_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// A tag keeps its address for the buffer's life (scatter programs embed it); invalidation only bumps gen.
constexpr uint64_t LATE_ALIGN = 256;
enum class LateRc { Ok, Unbound, Overrun, Regrow };

inline uint64_t lateAlignUp(uint64_t v) { return (v + LATE_ALIGN - 1) / LATE_ALIGN * LATE_ALIGN; }

// Bytes lateArenaAlloc needs for `blocks`; +LATE_ALIGN: the base is only 8-byte aligned.
inline uint64_t latePlanBytes(const std::vector<uint64_t> &blocks) {
    uint64_t a = 0;
    for (uint64_t v : blocks) a += lateAlignUp(v);
    return a + LATE_ALIGN;
}

struct LateArenaState {
    uint8_t *base = nullptr;
    uint64_t cap = 0, used = 0;
    uint32_t gen = 0;
    std::map<std::string, std::pair<uint64_t, uint64_t>> tags;  // tag -> (offset, reserved bytes)
};

inline std::mutex &lateArenaMutex() { static std::mutex m; return m; }
inline std::map<int, LateArenaState> &lateArenas() { static std::map<int, LateArenaState> m; return m; }

inline void lateArenaResetLocked(LateArenaState &a, void *base, uint64_t cap) {
    // An empty region stays unbound so the allocators keep their cudaMalloc fallback.
    a.base = cap == 0 ? nullptr : (uint8_t *)base;
    a.cap = a.base == nullptr ? 0 : cap;
    a.used = 0;
    a.tags.clear();
    ++a.gen;
}

inline void lateArenaBind(int gpuId, void *base, uint64_t cap) {
    std::lock_guard<std::mutex> lk(lateArenaMutex());
    LateArenaState &a = lateArenas()[gpuId];
    // Live tags mean caches still hold pointers into the old region; mul_release_device first.
    if (a.base != nullptr && !a.tags.empty()) {
        fprintf(stderr, "lateArenaBind: gpu %d is still bound with %zu tags; release it before rebinding\n", gpuId,
                a.tags.size());
        abort();
    }
    lateArenaResetLocked(a, base, cap);
}

inline void lateArenaUnbind(int gpuId) {
    std::lock_guard<std::mutex> lk(lateArenaMutex());
    lateArenaResetLocked(lateArenas()[gpuId], nullptr, 0);
}

inline void lateArenaInvalidate(int gpuId) {
    std::lock_guard<std::mutex> lk(lateArenaMutex());
    ++lateArenas()[gpuId].gen;
}

inline uint32_t lateArenaGen(int gpuId) {
    std::lock_guard<std::mutex> lk(lateArenaMutex());
    return lateArenas()[gpuId].gen;
}

inline LateRc lateArenaAlloc(int gpuId, const std::string &tag, uint64_t bytes, void **out) {
    std::lock_guard<std::mutex> lk(lateArenaMutex());
    LateArenaState &a = lateArenas()[gpuId];
    if (a.base == nullptr) return LateRc::Unbound;
    auto it = a.tags.find(tag);
    if (it != a.tags.end()) {
        if (bytes > it->second.second) return LateRc::Regrow;
        *out = a.base + it->second.first;
        return LateRc::Ok;
    }
    // Align the absolute address: the base itself is only 8-byte aligned.
    const uintptr_t abs = (uintptr_t)a.base + a.used;
    const uint64_t off = a.used + ((LATE_ALIGN - abs % LATE_ALIGN) % LATE_ALIGN);
    const uint64_t res = lateAlignUp(bytes);
    if (off > a.cap || res > a.cap - off) return LateRc::Overrun;
    a.tags[tag] = {off, res};
    a.used = off + res;
    *out = a.base + off;
    return LateRc::Ok;
}

inline uint64_t lateArenaUsed(int gpuId) {
    std::lock_guard<std::mutex> lk(lateArenaMutex());
    return lateArenas()[gpuId].used;
}

inline uint64_t lateArenaCap(int gpuId) {
    std::lock_guard<std::mutex> lk(lateArenaMutex());
    return lateArenas()[gpuId].cap;
}

inline bool lateArenaBound(int gpuId) {
    std::lock_guard<std::mutex> lk(lateArenaMutex());
    return lateArenas()[gpuId].base != nullptr;
}

// False for the cudaMalloc fallback's blocks.
inline bool lateArenaOwns(int gpuId, const void *p) {
    std::lock_guard<std::mutex> lk(lateArenaMutex());
    const LateArenaState &a = lateArenas()[gpuId];
    return a.base != nullptr && (const uint8_t *)p >= a.base && (const uint8_t *)p < a.base + a.cap;
}

#endif
