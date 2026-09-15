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

// Range-check tables only: a virtual range table addresses row `value + bias`, bias being -min for
// a specified range and 0 for a predefined one. Richer table families are out of scope.
struct MulDecoder {
    uint32_t table_id = 0;
    uint64_t acc_base = 0;
    uint64_t n_rows   = 0;
    int64_t  bias     = 0;          // -min, from the range_def hint
};

// Mirrors std_sum.pil: only the assumes side carries the tuple a lookup consumes.
#define MUL_PIOP_ASSUMES 0

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
