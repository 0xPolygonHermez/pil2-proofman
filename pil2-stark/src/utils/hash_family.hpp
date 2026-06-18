#ifndef HASH_FAMILY_HPP
#define HASH_FAMILY_HPP

#include <cstdint>
#include <stdexcept>
#include <string>

enum class HashFamily : uint8_t {
    Undefined = 0,
    Poseidon1 = 1,
    Poseidon2 = 2,
};

inline HashFamily parseHashFamily(const std::string &id) {
    if (id == "Poseidon1") return HashFamily::Poseidon1;
    if (id == "Poseidon2") return HashFamily::Poseidon2;
    throw std::runtime_error("unknown hash family: " + id);
}

inline const char *toString(HashFamily h) {
    switch (h) {
        case HashFamily::Poseidon1: return "Poseidon1";
        case HashFamily::Poseidon2: return "Poseidon2";
        case HashFamily::Undefined: return "Undefined";
    }
    return "Unknown";
}

// Only the two real families cross the FFI; Undefined is purely internal.
inline HashFamily hashFamilyFromU8(uint8_t v) {
    switch (v) {
        case static_cast<uint8_t>(HashFamily::Poseidon1): return HashFamily::Poseidon1;
        case static_cast<uint8_t>(HashFamily::Poseidon2): return HashFamily::Poseidon2;
        default: throw std::runtime_error("invalid HashFamily ABI value: " + std::to_string(v));
    }
}

// Process-global Poseidon family.
//
// `inline` only deduplicates within one linked image — each main binary or
// dlopen'd `.so` that links libstarksgpu.a gets its own copy. Every image
// that calls `get_hash_family` must first call `set_hash_family_c` once; the
// Undefined sentinel turns a missed init into a runtime error instead of a
// silent Poseidon2 default.
inline HashFamily g_hash_family = HashFamily::Undefined;

inline HashFamily get_hash_family() {
    if (g_hash_family == HashFamily::Undefined) {
        throw std::runtime_error(
            "get_hash_family: hash family not set in this linked image. "
            "Call set_hash_family at startup (GlobalInfo::load on the main "
            "binary; the WitnessComponent entry point on a witness library).");
    }
    return g_hash_family;
}

inline void define_hash_family(HashFamily f) {
    if (g_hash_family == f) return;
    if (g_hash_family != HashFamily::Undefined) {
        throw std::runtime_error(
            std::string("define_hash_family: cannot change hash family at runtime "
                        "(currently ") + toString(g_hash_family) +
            ", attempted " + toString(f) + ")");
    }
    g_hash_family = f;
}

#endif
