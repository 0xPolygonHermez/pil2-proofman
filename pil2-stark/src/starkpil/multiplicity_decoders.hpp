#ifndef MULTIPLICITY_DECODERS_HPP
#define MULTIPLICITY_DECODERS_HPP

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

// Compiled by BOTH nvcc (device scatter) and plain g++ (CPU scatter, unit test).
// __host__ __device__ is not valid outside nvcc.
#ifdef __CUDACC__
#define MUL_HD __host__ __device__
#else
#define MUL_HD
#endif

// How a lookup's tuple maps to the row of the table it addresses. A range table uses a single
// element at row `value + bias` (bias = -min, or 0 for a predefined range). A K-element tuple
// uses an affine fit derived from the table's COL_* columns, an exact map, or a digit rule.
#define MUL_MAX_TUPLE 8

struct MulDecoder {
    uint32_t table_id = 0;
    uint64_t acc_base = 0;
    uint64_t n_rows   = 0;
    int64_t  bias     = 0;          // -min, from the range_def hint (single-element tables)
    // Fitted map: row = sum(coef[i] * tuple[i]) + konst in the field. nCoef == 0: use `bias`.
    uint8_t  nCoef    = 0;
    uint64_t coef[MUL_MAX_TUPLE] = {0};
    uint64_t konst    = 0;
    // Exact-match map, for a table with no closed form: open addressing over slots of
    // [key columns..., row]. The key is the lookup's own tuple elements, stored and compared
    // VERBATIM -- no packing, so a column holding a negative field element (near p) or a full
    // 32-bit value is no obstacle, and a tuple the table does not hold misses cleanly instead of
    // landing on a wrong row.
    uint32_t nKey     = 0;              // key columns per slot; 0 = no map
    uint64_t mapSlots = 0;
    const uint64_t* mapKV = nullptr;    // mapSlots * (nKey + 1)
    // Separable row map: row = sum over the columns it uses of sum_i tab[(c*digits + i)*base + d].
    // Between the affine fit and the map -- 126's row is its packed column re-based and 125's is
    // affine in most columns and table-driven in one, neither of which is affine but neither of
    // which needs a map. `digitBase == 0` means "not a separable table".
    // `digitCols == 0` means "not a separable table". The per-column base, digit count and table
    // offsets ride in digitTab's own header -- `[base..][ndig..][off..][contributions..]` -- rather
    // than here, so the scatter job that carries this stays small: register pressure in that kernel
    // caps occupancy for every air, not just the tables that use a rule.
    uint32_t digitCols  = 0;
    uint32_t digitCol[MUL_MAX_TUPLE] = {0};   // which tuple element each rule column is
    const uint64_t* digitTab = nullptr;
    // Indexed-base row map: row = base[idx] + sum_c stride[c] * key[c], where idx packs the
    // selector bit-fields in order. Table 125 is 134 live bases plus two strides (A:1, B:256):
    // the block a tuple lands in is chosen by (OP, POS_IND, CIN, bit 4 of FLAGS), and every block
    // is exactly 2^16 rows with A and B uniform across the whole table.
    // `nSel == 0` means "not an indexed-base table".
    uint32_t nSel = 0;
    uint32_t selCol[MUL_MAX_TUPLE]   = {0};
    uint32_t selShift[MUL_MAX_TUPLE] = {0};
    uint64_t selMask[MUL_MAX_TUPLE]  = {0};
    const uint64_t* baseTab = nullptr;
    uint32_t nStride = 0;
    uint32_t strideCol[MUL_MAX_TUPLE] = {0};
    uint64_t strideVal[MUL_MAX_TUPLE] = {0};
};

// Populate a MulDecoder's indexed-base fields from the flat arrays `mul_register_table_indexed_base`
// carries: `sel` is (col, shift, mask) triples, `stride` is (col, stride) pairs, `base` is the flat
// table the packed selector addresses. `nBase` is accepted for symmetry with the registration entry
// point but not re-validated here -- the caller already checked it against the selector space.
inline void mulSetIndexedBase(MulDecoder& d, const uint64_t* sel, uint64_t nSel,
                              const uint64_t* base, uint64_t nBase,
                              const uint64_t* stride, uint64_t nStride) {
    (void)nBase;
    d.nSel = (uint32_t)nSel;
    for (uint64_t i = 0; i < nSel && i < MUL_MAX_TUPLE; ++i) {
        d.selCol[i]   = (uint32_t)sel[3 * i];
        d.selShift[i] = (uint32_t)sel[3 * i + 1];
        d.selMask[i]  = sel[3 * i + 2];
    }
    d.baseTab = base;
    d.nStride = (uint32_t)nStride;
    for (uint64_t i = 0; i < nStride && i < MUL_MAX_TUPLE; ++i) {
        d.strideCol[i] = (uint32_t)stride[2 * i];
        d.strideVal[i] = stride[2 * i + 1];
    }
}

#define MUL_DIGIT_INVALID 0xFFFFFFFFFFFFFFFFULL

#define MUL_MAP_EMPTY 0xFFFFFFFFFFFFFFFFULL

// Header word of a map whose keys are packed into one word. Must match MUL_MAP_PACKED in
// std_virtual_table.rs.
#define MUL_MAP_PACKED 0x5041434B45444B56ULL

// Longest probe chain a lookup may walk before it is treated as absent.
#define MUL_MAP_MAX_PROBE 4096ULL

// A key that lands outside the table.
#define MUL_INDEX_NONE 0xFFFFFFFFu

// Mirrors std_sum.pil: only the assumes side carries the tuple a lookup consumes.
#define MUL_PIOP_ASSUMES 0

// Key -> row. With no index the key IS the row (the range-check and affine-fit cases); with one,
// the key selects a row through the table's own index. Returns false when the key is not in the
// table, which the caller records as an out-of-range decode.
// Key -> row. With no map the key IS the row (the affine case); with one, the tuple selects a row
// through the table's own entries. Returns false when the tuple is not in the table, which the
// caller records as an out-of-range decode.
MUL_HD inline bool mulResolveRow(const uint64_t* key, uint32_t nKey, uint64_t mapSlots,
                                 const uint64_t* mapKV, uint64_t& row,
                                 uint32_t digitCols = 0, const uint64_t* digitTab = nullptr,
                                 const MulDecoder* d = nullptr) {
    if (d != nullptr && d->nSel != 0) {
        // idx packs the selector bit-fields, most significant first; row is the block's base plus
        // the strided columns (A, B for table 125). An idx the table never assigned a block --
        // rather than one we simply never counted -- is marked MUL_DIGIT_INVALID at registration, so
        // it misses cleanly here too.
        uint64_t idx = 0;
        for (uint32_t i = 0; i < d->nSel; ++i) {
            const uint64_t v = (key[d->selCol[i]] >> d->selShift[i]) & d->selMask[i];
            idx = idx * (d->selMask[i] + 1) + v;
        }
        const uint64_t b = d->baseTab[idx];
        if (b == MUL_DIGIT_INVALID) return false;
        uint64_t sum = b;
        for (uint32_t c = 0; c < d->nStride; ++c) sum += d->strideVal[c] * key[d->strideCol[c]];
        row = sum;
        return true;
    }
    if (digitCols != 0) {
        // The key holds exactly the columns the rule reads, in its order.
        uint64_t sum = 0;
        for (uint32_t c = 0; c < digitCols; ++c) {
            const uint64_t base = digitTab[c];
            const uint64_t nd   = digitTab[digitCols + c];
            // Malformed digit table: miss instead of dividing by zero.
            if (base < 2 || nd == 0) return false;
            const uint64_t* tab = digitTab + digitTab[2 * digitCols + c];
            uint64_t v = key[c];
            for (uint64_t i = 0; i < nd; ++i) {
                const uint64_t e = tab[i * base + (v % base)];
                if (e == MUL_DIGIT_INVALID) return false;
                v /= base;
                sum += e;
            }
            if (v != 0) return false;   // value wider than the table's digits
        }
        row = sum;
        return true;
    }
    if (mapSlots == 0) { row = key[0]; return true; }
    if (nKey == 0) return false;

    // Packed keys: the tuple fits in one word, so a slot is two words and the probe compares one.
    // The shape lives in the map's header.
    if (mapKV[0] == MUL_MAP_PACKED) {
        uint64_t packed = 0;
        for (uint32_t c = 0; c < nKey; ++c) {
            const uint64_t sw = mapKV[1 + c];
            const uint32_t sh = (uint32_t)(sw >> 32), w = (uint32_t)sw;
            const uint64_t v = key[c];
            // Wider than the column: not in the table (and would alias another key).
            if (w < 64 && (v >> w) != 0) return false;
            packed |= v << sh;
        }
        uint64_t h = packed + 0x9E3779B97F4A7C15ULL;
        h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 32;
        const uint64_t* kv = mapKV + 1 + nKey;
        uint64_t i = h & (mapSlots - 1);
        // Bounded so a fault cannot hang the stream; a miss becomes an out-of-range decode.
        const uint64_t limit = mapSlots < MUL_MAP_MAX_PROBE ? mapSlots : MUL_MAP_MAX_PROBE;
        for (uint64_t probe = 0; probe < limit; ++probe) {
            const uint64_t* slot = kv + i * 2;
            if (slot[0] == MUL_MAP_EMPTY) return false;
            if (slot[0] == packed) { row = slot[1]; return true; }
            i = (i + 1) & (mapSlots - 1);
        }
        return false;
    }      // a map without key columns cannot address anything
    const uint32_t stride = nKey + 1;
    uint64_t h = 0;
    for (uint32_t c = 0; c < nKey; ++c) {
        h ^= key[c] + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 32;
    }
    uint64_t i = h & (mapSlots - 1);
    // Bounded, as in the packed probe.
    const uint64_t limit = mapSlots < MUL_MAP_MAX_PROBE ? mapSlots : MUL_MAP_MAX_PROBE;
    for (uint64_t probe = 0; probe < limit; ++probe) {
        const uint64_t* slot = mapKV + i * stride;
        if (slot[0] == MUL_MAP_EMPTY) return false;          // absent
        bool hit = true;
        for (uint32_t c = 0; c < nKey; ++c) if (slot[c] != key[c]) { hit = false; break; }
        if (hit) { row = slot[nKey]; return true; }
        i = (i + 1) & (mapSlots - 1);
    }
    return false;
}

// {count, table_id, air, row} for the first out-of-table decode; slots 4..14 hold up to 11 key
// elements and slot 15 their count.
#define MUL_OOB_SLOTS 16

constexpr uint64_t MUL_P = 0xFFFFFFFF00000001ULL;   // Goldilocks

// Goldilocks values reaching here are not guaranteed reduced into [0, p).
MUL_HD inline uint64_t mulCanonHD(uint64_t v) { while (v >= MUL_P) v -= MUL_P; return v; }

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
// `key` is the full tuple that missed (or just the resolved row, for the affine case).

__device__ __forceinline__ void mulRecordOob(uint64_t* oob, uint32_t tableId, uint64_t air,
                                             const uint64_t* key, uint32_t nKey) {
    if (oob == nullptr) return;
    unsigned long long* o = (unsigned long long*)oob;
    if (atomicAdd(o, 1ULL) == 0) {
        o[1] = tableId; o[2] = air; o[3] = key[0];
        const uint32_t cap = MUL_OOB_SLOTS - 5;   // slots 4..14; slot 15 holds the count below
        const uint32_t n = nKey < cap ? nKey : cap;
        for (uint32_t c = 0; c < n; ++c) o[4 + c] = key[c];
        o[MUL_OOB_SLOTS - 1] = n;
    }
}
#endif

// Subtraction, for the compiled-program evaluator: the closed form never needed it (it folds
// signs into coefficients on the host), so these lived host-side in multiplicity_linear.hpp.
MUL_HD inline uint64_t mulNegFEHD(uint64_t a) { return a == 0 ? 0 : MUL_P - a; }
MUL_HD inline uint64_t mulSubFEHD(uint64_t a, uint64_t b) { return mulAddFE(a, mulNegFEHD(b)); }

MUL_HD inline uint64_t mul_decode(const MulDecoder& d, uint64_t value) {
    return mulAddFE(mulCanonHD(value), mulBiasFE(d.bias));
}

inline std::vector<MulDecoder>& mulDecoders() {
    static std::vector<MulDecoder> v;
    return v;
}

inline const MulDecoder* mulDecoderFor(uint64_t table_id) {
    for (const auto& d : mulDecoders()) if (d.table_id == table_id) return &d;
    return nullptr;
}

// Each row of the table must decode back to its own index: a stale bias fails here rather than
// silently miscounting.
inline bool mul_decoder_selfcheck(const MulDecoder& d, const uint64_t* column,
                                  uint64_t n_rows, std::string& err) {
    for (uint64_t r = 0; r < n_rows; ++r) {
        const uint64_t got = mul_decode(d, column[r]);
        if (got != r) {
            err = "decoder for table " + std::to_string(d.table_id) + " maps row "
                + std::to_string(r) + " to " + std::to_string(got);
            return false;
        }
    }
    return true;
}

#endif
