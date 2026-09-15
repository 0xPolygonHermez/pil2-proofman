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

// How a lookup's tuple maps to the row of the table it addresses.
//
// A range table is the degenerate case: a single-element tuple at row `value + bias`, bias being
// -min for a specified range and 0 for a predefined one. A general virtual table is addressed by a
// K-element tuple; when its layout is a packing, the row is affine in that tuple and the setup can
// FIT that map from the table's own COL_* fixed columns and verify it on every row. Anything that
// does not fit keeps nCoef == 0 and is left to whoever counted it before.
//
// Nothing here is table-specific: the coefficients are derived, never written down.
#define MUL_MAX_TUPLE 8

// Widest base a digit remap may use; the map is carried inline, so this bounds the job struct.
#define MUL_MAX_DIGIT_BASE 64

struct MulDecoder {
    uint32_t table_id = 0;
    uint64_t acc_base = 0;
    uint64_t n_rows   = 0;
    int64_t  bias     = 0;          // -min, from the range_def hint (single-element tables)
    // Fitted tuple map: row = sum(coef[i] * tuple[i]) + konst, as field elements. nCoef == 0 means
    // "not fitted", and the single-element `bias` path applies instead.
    uint8_t  nCoef    = 0;
    uint64_t coef[MUL_MAX_TUPLE] = {0};
    uint64_t konst    = 0;
    // Direct index, for a table whose row is NOT affine in its tuple but whose tuple does admit a
    // linear key that separates every row. The fitted coefficients above then produce that KEY, and
    // this turns the key into the row: `row = index[key - keyMin]`. Built from the table's own
    // fixed columns and verified there, exactly like the affine fit. indexLen == 0 means no index,
    // and the key IS the row.
    uint64_t keyMin   = 0;
    uint64_t indexLen = 0;
    const uint32_t* index = nullptr;   // host copy; the device copy is patched per GPU
    // Digit remap: the same relation an index expresses, when it happens to be a change of base.
    // `row = sum_i map[digit_i(key, baseIn)] * baseOut^i`. Keccak's chi table is exactly this
    // (base 28 in, base 16 out, six digits), so a 137 MB index collapses to a handful of bytes.
    // nDigits == 0 means no remap.
    uint32_t baseIn   = 0;
    uint32_t baseOut  = 0;
    uint32_t nDigits  = 0;
    uint32_t digitMap[MUL_MAX_DIGIT_BASE] = {0};
};

// A key that lands outside the table.
#define MUL_INDEX_NONE 0xFFFFFFFFu

// Mirrors std_sum.pil: only the assumes side carries the tuple a lookup consumes.
#define MUL_PIOP_ASSUMES 0

// Key -> row. With no index the key IS the row (the range-check and affine-fit cases); with one,
// the key selects a row through the table's own index. Returns false when the key is not in the
// table, which the caller records as an out-of-range decode.
struct MulRowMap {
    uint64_t keyMin;
    uint64_t indexLen;
    const uint32_t* index;
    uint32_t baseIn, baseOut, nDigits;
    const uint32_t* digitMap;
};

MUL_HD inline bool mulResolveRow(uint64_t key, const MulRowMap& m, uint64_t& row) {
    // A change of base, when the table's layout is one: no memory at all.
    if (m.nDigits != 0) {
        // 32-bit where the key allows it: a 64-bit divide costs several times a 32-bit one on the
        // device, and every table fitted so far has keys well inside 32 bits (chi's are < 28^6).
        if ((key >> 32) == 0) {
            uint32_t out = 0, mul = 1, k = (uint32_t)key;
            const uint32_t bi = m.baseIn, bo = m.baseOut;
            for (uint32_t i = 0; i < m.nDigits; ++i) {
                const uint32_t d = k % bi;
                k /= bi;
                const uint32_t mapped = m.digitMap[d];
                if (mapped == MUL_INDEX_NONE) return false;   // a digit the table never uses
                out += mapped * mul;
                mul *= bo;
            }
            if (k != 0) return false;
            row = out;
            return true;
        }
        uint64_t out = 0, mul = 1, k = key;
        for (uint32_t i = 0; i < m.nDigits; ++i) {
            const uint32_t d = (uint32_t)(k % m.baseIn);
            k /= m.baseIn;
            if (d >= MUL_MAX_DIGIT_BASE) return false;
            const uint32_t mapped = m.digitMap[d];
            if (mapped == MUL_INDEX_NONE) return false;
            out += (uint64_t)mapped * mul;
            mul *= m.baseOut;
        }
        if (k != 0) return false;                          // key wider than the fitted digits
        row = out;
        return true;
    }
    if (m.indexLen == 0) { row = key; return true; }
    const uint64_t off = key - m.keyMin;    // wraps below keyMin, caught by the bound
    if (off >= m.indexLen) return false;
    const uint32_t r = m.index[off];
    if (r == MUL_INDEX_NONE) return false;
    row = r;
    return true;
}

// {count, table_id, air, row} for the first decode that landed outside its table.
#define MUL_OOB_SLOTS 4

constexpr uint64_t MUL_P = 0xFFFFFFFF00000001ULL;   // Goldilocks

// Goldilocks values reaching here are not guaranteed reduced into [0, p).
MUL_HD inline uint64_t mulCanonHD(uint64_t v) { while (v >= MUL_P) v -= MUL_P; return v; }

// Bias applied in the field, not in int64: a range whose min is negative reaches the trace as a
// field element near p, so `value + (-min)` has to wrap through p to land on row 0.
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
// A correct decode never lands outside its table -- the lookup constrains the value. Record only
// the first, so a wrong decoder is diagnosable without syncing the stream.
__device__ __forceinline__ void mulRecordOob(uint64_t* oob, uint32_t tableId, uint64_t air, uint64_t idx) {
    if (oob == nullptr) return;
    unsigned long long* o = (unsigned long long*)oob;
    if (atomicAdd(o, 1ULL) == 0) { o[1] = tableId; o[2] = air; o[3] = idx; }
}
#endif

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
