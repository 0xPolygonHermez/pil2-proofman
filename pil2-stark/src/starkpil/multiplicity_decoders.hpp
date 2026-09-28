#ifndef MULTIPLICITY_DECODERS_HPP
#define MULTIPLICITY_DECODERS_HPP

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

// Compiled by BOTH nvcc (device scatter) and plain g++ (CPU scatter).
// __host__ __device__ is not valid outside nvcc.
#ifdef __CUDACC__
#define MUL_HD __host__ __device__
#else
#define MUL_HD
#endif

// How a lookup's tuple maps to the row of the table it addresses. A range table uses a single
// element at row `value + bias` (bias = -min, or 0 for a predefined range). A K-element tuple
// uses an exact map derived from the table's COL_* columns; K is unbounded.

struct MulDecoder {
    uint32_t table_id = 0;
    uint64_t hostAirKey = 0;        // mulAirKey of the virtual-table air holding the counters
    uint64_t acc_base = 0;
    uint64_t n_rows   = 0;
    int64_t  bias     = 0;          // -min, from the range_def hint (single-element tables)
    // Exact-match map (layout in std_virtual_table.rs); null for a range table.
    uint32_t nKey     = 0;              // key columns; 0 = no map
    uint64_t mapSlots = 0;
    const uint64_t* mapKV = nullptr;
};

#define MUL_MAP_EMPTY 0xFFFFFFFFFFFFFFFFULL

// Header word 0 of an exact map ("PACKKEY2"), then the shape and the slot count. Must match
// std_virtual_table.rs, which documents the layout.
#define MUL_MAP_MAGIC 0x5041434B4B455932ULL
#define MUL_MAP_HEAD 3
#define MUL_MAP_MAX_WORDS 8

// Longest probe chain a lookup may walk before it is treated as absent.
#define MUL_MAP_MAX_PROBE 4096ULL

// Mirrors std_sum.pil: only the assumes side carries the tuple a lookup consumes; the proves side
// is the table itself.
#define MUL_PIOP_ASSUMES 0
#define MUL_PIOP_PROVES 1

MUL_HD inline uint32_t mulMapWords(const uint64_t* m) { return (uint32_t)(m[1] >> 16) & 0xFF; }

// Fold key column `c`'s value into the packed key `kw` (zeroed by the caller). False: the value is
// outside the column's range, so the tuple is not in the table (and must not alias another key).
MUL_HD inline bool mulMapPack(const uint64_t* m, uint32_t c, uint64_t v, uint64_t* kw) {
    const uint64_t min = m[MUL_MAP_HEAD + 2 * c], meta = m[MUL_MAP_HEAD + 2 * c + 1];
    const uint32_t w = (uint32_t)meta & 0xFF, sh = (uint32_t)(meta >> 32) & 0xFF;
    if (v < min) return false;
    const uint64_t d = v - min;
    if (w < 64 && (d >> w) != 0) return false;
    kw[(meta >> 40) & 0xFF] |= d << sh;
    return true;
}

MUL_HD inline uint64_t mulMapHash(const uint64_t* kw, uint32_t nWords) {
    uint64_t h = 0;
    for (uint32_t w = 0; w < nWords; ++w) {
        h ^= kw[w] + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 32;
    }
    return h;
}

// Packed key -> row. Linear probing from (hash * slots) >> 64, bounded so a fault cannot hang the
// stream; a miss becomes an out-of-range decode.
MUL_HD inline bool mulMapFind(const uint64_t* m, const uint64_t* kw, uint64_t& row) {
    const uint64_t shape = m[1];
    const uint32_t nKey = (uint32_t)shape & 0xFFFF;
    const uint32_t nWords = (uint32_t)(shape >> 16) & 0xFF, slotWords = (uint32_t)(shape >> 24) & 0xFF;
    const uint32_t rowWord = (uint32_t)(shape >> 32) & 0xFF, rowShift = (uint32_t)(shape >> 40) & 0xFF;
    const bool inl = ((shape >> 48) & 1) != 0;
    const uint64_t slots = m[2];
    const uint64_t* tab = m + MUL_MAP_HEAD + 2 * nKey;
    const uint64_t keyMask = inl ? (1ULL << rowShift) - 1 : ~0ULL;
    uint64_t i = (uint64_t)(((unsigned __int128)mulMapHash(kw, nWords) * slots) >> 64);
    const uint64_t limit = slots < MUL_MAP_MAX_PROBE ? slots : MUL_MAP_MAX_PROBE;
    for (uint64_t probe = 0; probe < limit; ++probe) {
        const uint64_t* s = tab + i * slotWords;
        if (s[0] == MUL_MAP_EMPTY) return false;
        bool hit = true;
        for (uint32_t w = 0; w < nWords; ++w)
            if ((w == rowWord ? s[w] & keyMask : s[w]) != kw[w]) { hit = false; break; }
        if (hit) { row = inl ? s[rowWord] >> rowShift : s[nWords]; return true; }
        if (++i == slots) i = 0;
    }
    return false;
}

// {count, table_id, air, row} for the first out-of-table decode; slots 4..14 hold up to 11 key
// elements and slot 15 their count.
#define MUL_OOB_SLOTS 16

// A selector is a repetition count, added as an integer; one this large is a field value (e.g. -1)
// and is recorded as a bad lookup, its air word tagged with MUL_OOB_SELECTOR.
#define MUL_SEL_MAX (1ULL << 32)
#define MUL_OOB_SELECTOR (1ULL << 63)

constexpr uint64_t MUL_P = 0xFFFFFFFF00000001ULL;   // Goldilocks

// Goldilocks values reaching here are not guaranteed reduced into [0, p).
// 2p > 2^64, so one subtraction reaches the canonical form.
MUL_HD inline uint64_t mulCanonHD(uint64_t v) { return v >= MUL_P ? v - MUL_P : v; }

// Bias applied in the field: a negative min reaches the trace near p and must wrap to row 0.
MUL_HD inline uint64_t mulBiasFE(int64_t bias) {
    return bias >= 0 ? (uint64_t)bias : MUL_P - (uint64_t)(-bias);
}
MUL_HD inline uint64_t mulMulFE(uint64_t a, uint64_t b) {
    // Goldilocks reduce128: p = 2^64 - 2^32 + 1, so 2^64 == 2^32 - 1 (mod p). Inputs canonical.
    const unsigned __int128 r = (unsigned __int128)a * (unsigned __int128)b;
    const uint64_t lo = (uint64_t)r, hi = (uint64_t)(r >> 64);
    const uint64_t hi_hi = hi >> 32, hi_lo = hi & 0xFFFFFFFFULL;
    uint64_t t0 = lo - hi_hi;
    if (lo < hi_hi) t0 += MUL_P;
    const uint64_t t1 = hi_lo * 0xFFFFFFFFULL;
    uint64_t t2 = t0 + t1;
    if (t2 < t0) t2 += 0xFFFFFFFFULL;
    return t2 >= MUL_P ? t2 - MUL_P : t2;
}
MUL_HD inline uint64_t mulAddFE(uint64_t a, uint64_t b) {
    unsigned __int128 s = (unsigned __int128)a + b;
    return (uint64_t)(s >= MUL_P ? s - MUL_P : s);
}

#ifdef __CUDACC__
// A correct decode never lands outside its table. Record only the first, without syncing.
// `key` is the full tuple that missed (or just the resolved row, for a range table).
// Returns the record to the first caller only; it fills the key (slots 4..14) and its count
// (slot 15).
__device__ __forceinline__ unsigned long long* mulClaimOob(uint64_t* oob, uint32_t tableId,
                                                          uint64_t air, uint64_t first) {
    if (oob == nullptr) return nullptr;
    unsigned long long* o = (unsigned long long*)oob;
    if (atomicAdd(o, 1ULL) != 0) return nullptr;
    o[1] = tableId; o[2] = air; o[3] = first;
    return o;
}
__device__ __forceinline__ void mulRecordOob(uint64_t* oob, uint32_t tableId, uint64_t air,
                                             uint64_t row) {
    if (unsigned long long* o = mulClaimOob(oob, tableId, air, row)) {
        o[4] = row;
        o[MUL_OOB_SLOTS - 1] = 1;
    }
}
#endif

// Subtraction, for the compiled-program evaluator.
MUL_HD inline uint64_t mulNegFEHD(uint64_t a) { return a == 0 ? 0 : MUL_P - a; }
MUL_HD inline uint64_t mulSubFEHD(uint64_t a, uint64_t b) { return mulAddFE(a, mulNegFEHD(b)); }

inline std::vector<MulDecoder>& mulDecoders() {
    static std::vector<MulDecoder> v;
    return v;
}

inline const MulDecoder* mulDecoderFor(uint64_t table_id) {
    for (const auto& d : mulDecoders()) if (d.table_id == table_id) return &d;
    return nullptr;
}

#endif
