#ifndef POSEIDON_GOLDILOCKS_CUH
#define POSEIDON_GOLDILOCKS_CUH

// ---------------------------------------------------------------------------
// Poseidon v1 — GPU implementation.
// ---------------------------------------------------------------------------

#include "gl64_t.cuh"
#include "goldilocks_tooling.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "poseidon_goldilocks.hpp"
#include "goldilocks_trace_layout.cuh"
#include "poseidon_gpu_common.cuh"  // Layout, pow7(gl64_t&), scratchpad

#include "grinding_launch.hpp"

// Grinding launch geometry for POSEIDON1.
#define POSEIDON1_GRIND_BITS   19
#define POSEIDON1_GRIND_BLOCKS 512
#define POSEIDON1_GRIND_GRID \
    ((((1ULL << POSEIDON1_GRIND_BITS) + POSEIDON1_GRIND_BLOCKS - 1) / POSEIDON1_GRIND_BLOCKS))

static_assert(POSEIDON1_GRIND_GRID <= GRIND_NONCE_BLOCKS_MAX,
              "POSEIDON1 grinding grid exceeds the reserved nonce_blocks region");

// ---------------------------------------------------------------------------
// Public class (W=12).
// ---------------------------------------------------------------------------


template<uint32_t SPONGE_WIDTH_T>
class PoseidonGoldilocksGPU
{
public:
    static_assert(SPONGE_WIDTH_T == 8 || SPONGE_WIDTH_T == 12 || SPONGE_WIDTH_T == 16,
                  "PoseidonGoldilocksGPU: SPONGE_WIDTH_T must be 8, 12, or 16");

    static constexpr uint32_t CAPACITY            = 4;
    static constexpr uint32_t RATE                = SPONGE_WIDTH_T - CAPACITY;
    static constexpr uint32_t SPONGE_WIDTH        = SPONGE_WIDTH_T;
    static constexpr uint32_t N_FULL_ROUNDS_TOTAL = 8;
    static constexpr uint32_t HALF_N_FULL_ROUNDS  = 4;
    static constexpr uint32_t N_PARTIAL_ROUNDS    = PoseidonGoldilocksConstants::Poseidon1Tables<SPONGE_WIDTH_T>::N_PARTIAL_ROUNDS;
    static constexpr uint32_t N_ROUNDS            = N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS;

    static void initConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);

    static void permute(uint64_t *output, const uint64_t *input, cudaStream_t stream = 0);
    static void permuteTrunc(uint64_t *output, const uint64_t *input, cudaStream_t stream = 0);

    static void linearHash(uint64_t *d_hash_output, uint64_t *d_trace,
                           uint64_t num_cols, uint64_t num_rows,
                           Layout layout, cudaStream_t stream);

    static void merkletree(uint32_t arity, uint64_t *d_tree, uint64_t *d_input,
                           uint64_t num_cols, uint64_t num_rows,
                           Layout layout, cudaStream_t stream);

    static void merkletreeReduce(uint64_t *d_root, uint64_t *d_input,
                                 uint64_t num_elements, uint64_t arity,
                                 cudaStream_t stream);

    static void grinding(uint64_t *d_nonce, uint64_t *d_nonceBlock,
                         const uint64_t *d_in, const uint32_t n_bits,
                         cudaStream_t stream);
};

using PoseidonGoldilocksGPUGrinding = PoseidonGoldilocksGPU<8>;

// ---------------------------------------------------------------------------
// Lazy-reduction arithmetic
//
// The MDS matrices M_8/M_12/M_16 have tiny entries (<= 7 bits), yet the naive
// path pays a full 64x64 modular multiply + reduction per term. Instead we
// accumulate the raw products wide and do a single modular reduction per
// dot product:
//   - M (tiny constants): compile-time immediates, sum fits u128 -> reduce128
//     (see the pos1_*_mimm_ helpers below).
//   - full-width constants (P, S): 16 x 128-bit products need a carry limb
//     (sum < 2^132) -> reduce160 (handles up to c*2^128 + 128 bits, c < 2^32),
//     using 2^128 == -2^32 (mod p).
// The reductions produce canonical results; inputs may be any u64.
// ---------------------------------------------------------------------------

// 2^64 mod p = 2^32 - 1 (the "epsilon" of the Goldilocks field): the
// correction to apply whenever a value crosses the 2^64 boundary (borrow or
// carry) during reduction.
__device__ constexpr uint64_t POS1_EPSILON = 0xFFFFFFFFULL;

// (hi*2^64 + lo) mod p, canonical. hi = hh*2^32 + hl:
// v == lo - hh + hl*(2^32 - 1)  (mod p)
__device__ __forceinline__ uint64_t pos1_reduce128_(uint64_t hi, uint64_t lo)
{
    uint32_t hh = (uint32_t)(hi >> 32);
    uint32_t hl = (uint32_t)hi;
    uint64_t t0 = lo - hh;
    if (lo < hh) t0 -= POS1_EPSILON;          // borrow: the wrap added 2^64 == epsilon, take it back
    uint64_t t1 = ((uint64_t)hl << 32) - hl;  // hl * (2^32 - 1), fits u64
    uint64_t r = t0 + t1;
    if (r < t1) r += POS1_EPSILON;            // carry: the wrap dropped 2^64 == epsilon, restore it
    if (r >= GOLDILOCKS_PRIME) r -= GOLDILOCKS_PRIME;
    return r;
}

// (c*2^128 + hi*2^64 + lo) mod p, canonical, using 2^128 == -2^32 (mod p).
__device__ __forceinline__ uint64_t pos1_reduce160_(uint32_t c, uint64_t hi, uint64_t lo)
{
    uint64_t r = pos1_reduce128_(hi, lo);
    uint64_t sub = ((uint64_t)c) << 32;
    r = (r >= sub) ? (r - sub) : (r + GOLDILOCKS_PRIME - sub);
    if (r >= GOLDILOCKS_PRIME) r -= GOLDILOCKS_PRIME;
    return r;
}

// 3-limb carry-chained MAC: (cy:hi:lo) += a*b in one hardware carry chain
// (mad.lo.cc / madc.hi.cc / addc = 3 instructions). The compiler's u128 +
// compare-based carry tracking costs ~2x that plus register-pair marshalling
// MOVs; the value semantics are identical.
__device__ __forceinline__ void pos1_mac_cc_(uint64_t &lo, uint64_t &hi, uint32_t &cy,
                                             uint64_t a, uint64_t b)
{
    asm("mad.lo.cc.u64 %0, %3, %4, %0;\n\t"
        "madc.hi.cc.u64 %1, %3, %4, %1;\n\t"
        "addc.u32 %2, %2, 0;"
        : "+l"(lo), "+l"(hi), "+r"(cy)
        : "l"(a), "l"(b));
}

// Dot product with full-width constants: 3-limb carry-chained accumulation.
template<uint32_t W>
__device__ __forceinline__ uint64_t pos1_dot_wide_raw_(const uint64_t *s, const gl64_t *col, uint32_t stride)
{
    unsigned __int128 first = (unsigned __int128)s[0] * col[0][0];
    uint64_t lo = (uint64_t)first, hi = (uint64_t)(first >> 64);
    uint32_t cy = 0;
#pragma unroll
    for (uint32_t j = 1; j < W; ++j)
        pos1_mac_cc_(lo, hi, cy, s[j], col[j * stride][0]);
    return pos1_reduce160_(cy, hi, lo);
}

// x + a*b with a single reduction. Bound: (2^64-1)^2 + (2^64-1) < 2^128.
__device__ __forceinline__ uint64_t pos1_muladd_raw_(uint64_t x, uint64_t a, uint64_t b)
{
    unsigned __int128 v = (unsigned __int128)a * b + x;
    return pos1_reduce128_((uint64_t)(v >> 64), (uint64_t)v);
}

// MDS dot product with the M matrix as COMPILE-TIME immediates: the M
// entries are tiny (<= 7 bits) constexpr values, so with the row loop
// expanded via template recursion every term is a multiply by a small
// literal, which the compiler strength-reduces far below a runtime 64x64
// product (entries equal to 1 become plain adds). The entry is extracted
// through a constexpr function so it is frontend-evaluated -- no device-side
// access to the host constexpr array.
template<uint32_t W>
__host__ __device__ constexpr uint64_t pos1_m_entry_(uint32_t j, uint32_t i)
{
    return PoseidonGoldilocksConstants::Poseidon1Tables<W>::M[j][i].fe;
}

template<uint32_t W, uint32_t I, uint32_t J = 0>
__device__ __forceinline__ void pos1_dot_mimm_rows_(unsigned __int128 &acc, const uint64_t *s)
{
    if constexpr (J < W)
    {
        constexpr uint64_t m = pos1_m_entry_<W>(J, I);
        if constexpr (m == 1)
            acc += s[J];
        else if constexpr (m != 0)
            acc += (unsigned __int128)s[J] * m;
        pos1_dot_mimm_rows_<W, I, J + 1>(acc, s);
    }
}

template<uint32_t W, uint32_t I>
__device__ __forceinline__ uint64_t pos1_dot_mimm_(const uint64_t *s)
{
    unsigned __int128 acc = 0;
    pos1_dot_mimm_rows_<W, I>(acc, s);
    return pos1_reduce128_((uint64_t)(acc >> 64), (uint64_t)acc);
}

// Immediate-MDS x state, shared-memory state. out[I] = sum_j M[j][I]*s[j]:
// pos1_dot_mimm_rows_ recurses over the matrix rows J of one column, and
// the _cols_ helpers recurse over the columns I (one per output word) -- both
// compile-time so M[J][I] stays a literal.
template<uint32_t W, uint32_t I = 0>
__device__ __forceinline__ void pos1_mvp_smem_mimm_cols_(const uint64_t *s)
{
    if constexpr (I < W)
    {
        gl64_t r;
        r[0] = pos1_dot_mimm_<W, I>(s);
        scratchpad[I * blockDim.x + threadIdx.x] = r;
        pos1_mvp_smem_mimm_cols_<W, I + 1>(s);
    }
}

template<uint32_t W>
__device__ __forceinline__ void pos1_mvp_smem_mimm_()
{
    uint64_t s[W];
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        s[i] = scratchpad[i * blockDim.x + threadIdx.x][0];
    pos1_mvp_smem_mimm_cols_<W>(s);
}

// Immediate-MDS x state, register state.
template<uint32_t W, uint32_t I = 0>
__device__ __forceinline__ void pos1_mvp_mimm_cols_(gl64_t *state, const uint64_t *olds)
{
    if constexpr (I < W)
    {
        state[I][0] = pos1_dot_mimm_<W, I>(olds);
        pos1_mvp_mimm_cols_<W, I + 1>(state, olds);
    }
}

template<uint32_t W>
__device__ __forceinline__ void pos1_mvp_mimm_(gl64_t *state)
{
    uint64_t olds[W];
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        olds[i] = state[i][0];
    pos1_mvp_mimm_cols_<W>(state, olds);
}

template<uint32_t W>
__device__ __forceinline__ void pos1_pow7_(gl64_t *x)
{
    gl64_t x2[W], x3[W], x4[W];
#pragma unroll
    for (int i = 0; i < (int)W; ++i)
    {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i]  = x3[i] * x4[i];
    }
}

template<uint32_t W>
__device__ __forceinline__ void pos1_add_(gl64_t *x, const gl64_t *C)
{
#pragma unroll
    for (int i = 0; i < (int)W; ++i)
        x[i] = x[i] + C[i];
}

template<uint32_t W>
__device__ __forceinline__ void pos1_prod_(gl64_t *x, const gl64_t alpha, const gl64_t *C)
{
#pragma unroll
    for (int i = 0; i < (int)W; ++i)
        x[i] = alpha * C[i];
}

template<uint32_t W>
__device__ __forceinline__ void pos1_pow7add_(gl64_t *x, const gl64_t *C)
{
    gl64_t x2[W], x3[W], x4[W];
#pragma unroll
    for (int i = 0; i < (int)W; ++i)
    {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i]  = x3[i] * x4[i];
        x[i]  = x[i] + C[i];
    }
}

template<uint32_t W>
__device__ __forceinline__ gl64_t pos1_dot_(const gl64_t *x, const gl64_t *C)
{
    uint64_t s[W];
#pragma unroll
    for (int i = 0; i < (int)W; ++i)
        s[i] = x[i][0];
    gl64_t r;
    r[0] = pos1_dot_wide_raw_<W>(s, C, 1);
    return r;
}

template<uint32_t W>
__device__ __forceinline__ void pos1_mvp_(gl64_t *state, const gl64_t *mat)
{
    uint64_t old_state[W];
#pragma unroll
    for (int i = 0; i < (int)W; ++i)
        old_state[i] = state[i][0];

#pragma unroll 1
    for (int i = 0; i < (int)W; ++i)
        state[i][0] = pos1_dot_wide_raw_<W>(old_state, &mat[i], W);
}

// ---------------------------------------------------------------------------
// Full register-path permutation.
// ---------------------------------------------------------------------------

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__device__ void poseidon1PermuteReg(gl64_t *state,
                                    const gl64_t *input,
                                    const gl64_t *GPU_C_GL,
                                    const gl64_t *GPU_S_GL,
                                    const gl64_t *GPU_M_GL,
                                    const gl64_t *GPU_P_GL)
{

    mymemcpy((uint64_t *)state, (uint64_t *)input, W);

    // First half: ARK -> (pow7+ARK, MVP(M)) × (HALF_F - 1).
    pos1_add_<W>(state, GPU_C_GL);
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        pos1_pow7add_<W>(state, &GPU_C_GL[(r + 1) * W]);
        pos1_mvp_mimm_<W>(state);
    }

    // Transition into partial rounds: pow7+ARK, MVP(P).
    pos1_pow7add_<W>(state, &GPU_C_GL[HALF_F * W]);
    pos1_mvp_<W>(state, GPU_P_GL);

    // Partial rounds: dot + rank-1 update, both with lazy reduction.
    for (uint32_t r = 0; r < N_PART; ++r)
    {
        pow7(state[0]);
        state[0] = state[0] + GPU_C_GL[(HALF_F + 1) * W + r];

        const gl64_t *S_row = &GPU_S_GL[(W * 2 - 1) * r];
        gl64_t s0 = pos1_dot_<W>(state, S_row);

        uint64_t a = state[0][0];
#pragma unroll
        for (uint32_t j = 1; j < W; ++j)
            state[j][0] = pos1_muladd_raw_(state[j][0], a, S_row[W - 1 + j][0]);
        state[0] = s0;
    }

    // Second half: (pow7+ARK, MVP(M)) × (HALF_F - 1), then pow7 + MVP(M).
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        pos1_pow7add_<W>(state, &GPU_C_GL[(HALF_F + 1) * W + N_PART + r * W]);
        pos1_mvp_mimm_<W>(state);
    }
    pos1_pow7_<W>(state);
    pos1_mvp_mimm_<W>(state);

}

// ---------------------------------------------------------------------------
// Shared-memory variant of the permutation.
// State layout: scratchpad[i * blockDim.x + threadIdx.x] = state element i of
// this thread (one thread per sponge). Avoids the register path's stack spill,
// and fuses dot + rank-1 in the partial rounds.
// ---------------------------------------------------------------------------

template<uint32_t W>
__device__ __forceinline__ void pos1_pow7_smem_()
{
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
    {
        gl64_t x  = scratchpad[i * blockDim.x + threadIdx.x];
        gl64_t x2 = x * x;
        gl64_t x3 = x * x2;
        gl64_t x4 = x2 * x2;
        scratchpad[i * blockDim.x + threadIdx.x] = x3 * x4;
    }
}

template<uint32_t W>
__device__ __forceinline__ void pos1_add_smem_(const gl64_t *C)
{
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        scratchpad[i * blockDim.x + threadIdx.x] =
            scratchpad[i * blockDim.x + threadIdx.x] + C[i];
}

// Fused pow7 + ARK: state[i] = pow7(state[i]) + C[i], single read / single
// write per element (vs. pow7_smem_ + add_smem_ doing two passes).
template<uint32_t W>
__device__ __forceinline__ void pos1_pow7add_smem_(const gl64_t *C)
{
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
    {
        gl64_t x  = scratchpad[i * blockDim.x + threadIdx.x];
        gl64_t x2 = x * x;
        gl64_t x3 = x * x2;
        gl64_t x4 = x2 * x2;
        scratchpad[i * blockDim.x + threadIdx.x] = (x3 * x4) + C[i];
    }
}

// Fused full round: state[I] = sum_J M[J][I] * (pow7(state[J]) + C[J]), one
// smem read and one smem write per element per round (the split
// pow7add_smem_ + mvp_smem_mimm_ pair does two of each), with each lane's
// pow7 mul chain (fma pipe) interleaving against the previous lane's
// immediate-MDS accumulation adds (alu pipe). Accumulator bound: W * max(M)
// * 2^64 < 2^75 fits the u128 with no carry limb (static-asserted in the .cu).
// NOTE: deliberately plain u128 arithmetic, NOT the asm MAC helpers — the M
// entries are tiny compile-time constants the compiler strength-reduces onto
// the full-rate alu pipe (shifts/adds); forcing mad here moves the work onto
// the scarce half-rate mul pipe (measured +5% regression on W16).
template<uint32_t W, uint32_t J, uint32_t I = 0>
__device__ __forceinline__ void pos1_fused_scatter_(unsigned __int128 *acc, uint64_t t)
{
    if constexpr (I < W)
    {
        constexpr uint64_t m = pos1_m_entry_<W>(J, I);
        if constexpr (m == 1)
            acc[I] += t;
        else if constexpr (m != 0)
            acc[I] += (unsigned __int128)t * m;
        pos1_fused_scatter_<W, J, I + 1>(acc, t);
    }
}

template<uint32_t W, bool ARK, uint32_t J = 0>
__device__ __forceinline__ void pos1_fused_rows_(unsigned __int128 *acc, const gl64_t *C)
{
    if constexpr (J < W)
    {
        gl64_t x = scratchpad[J * blockDim.x + threadIdx.x];
        gl64_t x2 = x * x;
        gl64_t x3 = x * x2;
        gl64_t x4 = x2 * x2;
        gl64_t t = x3 * x4;
        uint64_t tj;
        if constexpr (ARK)
            tj = (t + C[J])[0];
        else
            tj = t[0];
        pos1_fused_scatter_<W, J>(acc, tj);
        pos1_fused_rows_<W, ARK, J + 1>(acc, C);
    }
}

// ARK=false is the schedule's final full round (pow7 + MDS, no constants);
// that instantiation never reads C.
#ifndef POS1_MDS_CIRC
#define POS1_MDS_CIRC 1
#endif
// ---------------------------------------------------------------------------
// Circulant fast-MDS full round (W=16). M16 is circulant: M[i][j] = k[(j-i)%16]
// with k = row 0, so the MDS product is a cyclic convolution out = k (*) t.
// CRT split of x^16-1 = (x^8-1)(x^8+1): fold tc[j]=t[j]+t[j+8], tn[j]=t[j]-t[j+8],
// evaluate two 8-point convolutions with the half-kernels kc = k_lo+k_hi (all
// positive, <= 102) and kn = k_lo-k_hi (max |tap| 100; after the negacyclic
// wraparound up to 7 of an output's 8 terms are negative -> compile-time signs
// with one p<<7 bias per negative term, see pos1_circ_bias_), and reconstruct
// out[i] = (U[i]+V[i])/2, out[i+8] = (U[i]-V[i])/2.
// U +/- V go through gl64_t ops, which under GL64_PARTIALLY_REDUCED may return
// values in [p, 2^64); pos1_halve_ is congruence-exact for any 64-bit input and
// canonicalizes, so the result is bit-identical to the direct dot product.
// Accumulator peaks: aU <= sum(kc)*(2^64-1) < 2^73, aV <= bias + positive
// terms < 2^74 -> u128, no carry limb (machine-checked against M by
// pos1_circ_bounds_ok_).
// Cost: 128 MACs + 16 folds + 16 reductions + 16 halvings vs 256 MACs + 16
// reductions.
// ---------------------------------------------------------------------------
template<uint32_t W>
__host__ __device__ constexpr bool pos1_is_circulant_()
{
    for (uint32_t i = 0; i < W; ++i)
        for (uint32_t j = 0; j < W; ++j)
            if (pos1_m_entry_<W>(i, j) != pos1_m_entry_<W>(0, (j + W - i) % W)) return false;
    return true;
}
template<uint32_t W>
__host__ __device__ constexpr uint64_t pos1_kc_(uint32_t j)
{
    return pos1_m_entry_<W>(0, j) + pos1_m_entry_<W>(0, j + W / 2);
}
template<uint32_t W>
__host__ __device__ constexpr int64_t pos1_kn_(uint32_t j)
{
    return (int64_t)pos1_m_entry_<W>(0, j) - (int64_t)pos1_m_entry_<W>(0, j + W / 2);
}

// Exact field halving, 2*r == v (mod p): even -> v/2; odd -> (v-1)/2 + (p+1)/2
// (== (v+p)/2, no 64-bit overflow). Correct for ANY 64-bit v, not just v < p:
// under GL64_PARTIALLY_REDUCED gl64_t add/sub can return values in [p, 2^64)
// (e.g. U = p-1, V = 1 -> U+V = p, unreduced), and an odd v >= p would yield
// r = (v+p)/2 >= p -- a non-canonical digest word escaping the final round.
// r <= p + 2^31 always, so one conditional subtract canonicalizes and keeps
// the output bit-identical to the direct MDS path.
__device__ __forceinline__ gl64_t pos1_halve_(gl64_t x)
{
    uint64_t v = x[0];
    uint64_t r = (v >> 1) + ((v & 1) ? ((GOLDILOCKS_PRIME + 1) >> 1) : 0ull);
    if (r >= GOLDILOCKS_PRIME) r -= GOLDILOCKS_PRIME;
    gl64_t g; g[0] = r; return g;
}

// Effective negacyclic sign of the (I, J) term: wraparound (J > I) times the
// kernel tap's own sign — both compile-time.
template<uint32_t W>
__host__ __device__ constexpr int64_t pos1_circ_sm_(uint32_t i, uint32_t j)
{
    const int64_t kv = pos1_kn_<W>((i + W / 2 - j) % (W / 2));
    return (j <= i) ? kv : -kv;
}
// One underflow cushion for the negacyclic accumulators: p<<7. It must cover
// the largest single negative term, max|kn|*(2^64-1) -- i.e. taps up to 127.
// That, and the aU/aV overflow headroom, are statically checked against the
// actual matrix by pos1_circ_bounds_ok_ below.
__host__ __device__ constexpr unsigned __int128 pos1_circ_cushion_()
{
    return (unsigned __int128)GOLDILOCKS_PRIME << 7;
}

// Compile-time bias for output I of the negacyclic half: one cushion per
// negative term, so the accumulator never underflows and stays == V[I] (mod p)
// with subtraction done directly in the 128-bit domain — one accumulator and
// one reduction instead of a P/N pair.
template<uint32_t W>
__host__ __device__ constexpr unsigned __int128 pos1_circ_bias_(uint32_t i)
{
    uint32_t cnt = 0;
    for (uint32_t j = 0; j < W / 2; ++j)
        if (pos1_circ_sm_<W>(i, j) < 0) ++cnt;
    return pos1_circ_cushion_() * cnt;
}

// Soundness of the cushion/accumulator scheme against the ACTUAL matrix
// coefficients (evaluated at compile time, asserted in pos1_round_fused_circ_):
//  1) one cushion covers any single negative term: |kn[j]|*(2^64-1) <= p<<7
//  2) aV never overflows: bias(i) + sum of positive-term maxima fits u128
//  3) aU never overflows: sum(kc)*(2^64-1) fits u128
// Everything derives from Poseidon1Tables<W>::M, so a regenerated or swapped
// constants table that breaks any bound fails the build instead of the proof.
template<uint32_t W>
__host__ __device__ constexpr bool pos1_circ_bounds_ok_()
{
    constexpr unsigned __int128 X = ~0ull;  // max raw 64-bit input
    constexpr unsigned __int128 U128_MAX = ~(unsigned __int128)0;
    unsigned __int128 sum_kc = 0;
    for (uint32_t j = 0; j < W / 2; ++j)
    {
        const int64_t kn = pos1_kn_<W>(j);
        const uint64_t mag = (uint64_t)(kn < 0 ? -kn : kn);
        if ((unsigned __int128)mag * X > pos1_circ_cushion_()) return false;
        sum_kc += pos1_kc_<W>(j);
    }
    if (sum_kc != 0 && sum_kc > U128_MAX / X) return false;
    for (uint32_t i = 0; i < W / 2; ++i)                                      // (2)
    {
        unsigned __int128 total = pos1_circ_bias_<W>(i);
        for (uint32_t j = 0; j < W / 2; ++j)
        {
            const int64_t sm = pos1_circ_sm_<W>(i, j);
            if (sm <= 0) continue;
            const unsigned __int128 t = (unsigned __int128)(uint64_t)sm * X;
            if (total > U128_MAX - t) return false;
            total += t;
        }
    }
    return true;
}

// Scatter one folded input pair (tc, tn from lane J) into the half-size
// accumulators; coefficients and negacyclic signs are compile-time.
template<uint32_t W, uint32_t J, uint32_t I = 0>
__device__ __forceinline__ void pos1_circ_scatter_(unsigned __int128 *aU,
                                                   unsigned __int128 *aV,
                                                   uint64_t tc, uint64_t tn)
{
    constexpr uint32_t H = W / 2;
    if constexpr (I < H)
    {
        constexpr uint64_t mc = pos1_kc_<W>((I + H - J) % H);
        if constexpr (mc == 1) aU[I] += tc; else aU[I] += (unsigned __int128)tc * mc;
        constexpr int64_t sm = pos1_circ_sm_<W>(I, J);
        if constexpr (sm > 0)
        {
            if constexpr (sm == 1) aV[I] += tn; else aV[I] += (unsigned __int128)tn * (uint64_t)sm;
        }
        else if constexpr (sm < 0)
        {
            if constexpr (sm == -1) aV[I] -= tn; else aV[I] -= (unsigned __int128)tn * (uint64_t)(-sm);
        }
        pos1_circ_scatter_<W, J, I + 1>(aU, aV, tc, tn);
    }
}

// Unrolled bias initialization (I is a template constant so the bias folds to
// two immediate words per accumulator).
template<uint32_t W, uint32_t I = 0>
__device__ __forceinline__ void pos1_circ_init_(unsigned __int128 *aV)
{
    if constexpr (I < W / 2)
    {
        aV[I] = pos1_circ_bias_<W>(I);
        pos1_circ_init_<W, I + 1>(aV);
    }
}

template<uint32_t W, bool ARK, uint32_t J = 0>
__device__ __forceinline__ void pos1_circ_rows_(unsigned __int128 *aU,
                                                unsigned __int128 *aV,
                                                const gl64_t *C)
{
    constexpr uint32_t H = W / 2;
    if constexpr (J < H)
    {
        gl64_t x = scratchpad[J * blockDim.x + threadIdx.x];
        gl64_t x2 = x * x; gl64_t x3 = x * x2; gl64_t x4 = x2 * x2;
        gl64_t ta = x3 * x4;
        gl64_t y = scratchpad[(J + H) * blockDim.x + threadIdx.x];
        gl64_t y2 = y * y; gl64_t y3 = y * y2; gl64_t y4 = y2 * y2;
        gl64_t tb = y3 * y4;
        if constexpr (ARK) { ta = ta + C[J]; tb = tb + C[J + H]; }
        gl64_t u = ta + tb;
        gl64_t v = ta - tb;
        pos1_circ_scatter_<W, J>(aU, aV, u[0], v[0]);
        pos1_circ_rows_<W, ARK, J + 1>(aU, aV, C);
    }
}

template<uint32_t W, bool ARK = true>
__device__ __forceinline__ void pos1_round_fused_circ_(const gl64_t *C = nullptr)
{
    static_assert(pos1_is_circulant_<W>(), "circulant MDS round requires a circulant M");
    static_assert(pos1_circ_bounds_ok_<W>(),
                  "circulant MDS round: p<<7 cushion / u128 headroom does not hold for this M");
    constexpr uint32_t H = W / 2;
    unsigned __int128 aU[H] = {}, aV[H];
    pos1_circ_init_<W>(aV);
    pos1_circ_rows_<W, ARK>(aU, aV, C);
#pragma unroll
    for (uint32_t i = 0; i < H; ++i)
    {
        gl64_t U; U[0] = pos1_reduce128_((uint64_t)(aU[i] >> 64), (uint64_t)aU[i]);
        gl64_t V; V[0] = pos1_reduce128_((uint64_t)(aV[i] >> 64), (uint64_t)aV[i]);
        scratchpad[i * blockDim.x + threadIdx.x]       = pos1_halve_(U + V);
        scratchpad[(i + H) * blockDim.x + threadIdx.x] = pos1_halve_(U - V);
    }
}

template<uint32_t W, bool ARK = true>
__device__ __forceinline__ void pos1_round_fused_smem_(const gl64_t *C = nullptr)
{
    unsigned __int128 acc[W] = {};
    pos1_fused_rows_<W, ARK>(acc, C);
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
    {
        gl64_t r;
        r[0] = pos1_reduce128_((uint64_t)(acc[i] >> 64), (uint64_t)acc[i]);
        scratchpad[i * blockDim.x + threadIdx.x] = r;
    }
}

// P x state in shared memory. Indexing matches pos1_mvp_ (mat[W*j + i],
// transposed). Outer loop kept at unroll-1 on purpose: fully unrolling it
// spikes register use and drops occupancy.
template<uint32_t W>
__device__ __forceinline__ void pos1_mvp_smem_(const gl64_t *mat)
{
    uint64_t s[W];
#pragma unroll
    for (uint32_t i = 0; i < W; ++i)
        s[i] = scratchpad[i * blockDim.x + threadIdx.x][0];

#pragma unroll 1
    for (uint32_t i = 0; i < W; ++i)
    {
        gl64_t r;
        r[0] = pos1_dot_wide_raw_<W>(s, &mat[i], W);
        scratchpad[i * blockDim.x + threadIdx.x] = r;
    }
}

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__device__ void poseidon1PermuteSmem(const gl64_t *GPU_C_GL,
                                     const gl64_t *GPU_S_GL,
                                     const gl64_t *GPU_M_GL,
                                     const gl64_t *GPU_P_GL)
{


    // First half: initial ARK, then (HALF_F - 1) × (pow7+ARK fused + MVP(M)).
    pos1_add_smem_<W>(GPU_C_GL);
#pragma unroll 1
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        // W=16 (production commit sponge, occupancy-limited) measures faster with
        // the fused round;
        if constexpr (W == 16)
        {
#if POS1_MDS_CIRC
            pos1_round_fused_circ_<W>(&GPU_C_GL[(r + 1) * W]);
#else
            pos1_round_fused_smem_<W>(&GPU_C_GL[(r + 1) * W]);
#endif
        }
        else
        {
            pos1_pow7add_smem_<W>(&GPU_C_GL[(r + 1) * W]);
            pos1_mvp_smem_mimm_<W>();
        }
    }

    // Transition full round → MVP(P).
    pos1_pow7add_smem_<W>(&GPU_C_GL[HALF_F * W]);
    pos1_mvp_smem_<W>(GPU_P_GL);

    // Partial rounds — fused dot + rank-1 loop, both lazily reduced. state[0]
    // is rewritten every round, so keep it in a register across all rounds
    // (read/write scratchpad once). Caching the full state in registers
    // instead spikes register use and drops occupancy.
    {
        gl64_t s0_reg = scratchpad[threadIdx.x];
#pragma unroll 1
        for (uint32_t r = 0; r < N_PART; ++r)
        {
            gl64_t x2 = s0_reg * s0_reg;
            gl64_t x3 = s0_reg * x2;
            gl64_t x4 = x2 * x2;
            s0_reg = (x3 * x4) + GPU_C_GL[(HALF_F + 1) * W + r];

            const gl64_t *S_row = &GPU_S_GL[(W * 2 - 1) * r];
            uint64_t a = s0_reg[0];
            unsigned __int128 first = (unsigned __int128)a * S_row[0][0];
            uint64_t lo = (uint64_t)first, hi = (uint64_t)(first >> 64);
            uint32_t cy = 0;
#pragma unroll
            for (uint32_t j = 1; j < W; ++j)
            {
                uint64_t sj = scratchpad[j * blockDim.x + threadIdx.x][0];
                pos1_mac_cc_(lo, hi, cy, sj, S_row[j][0]);
                gl64_t nsj;
                nsj[0] = pos1_muladd_raw_(sj, a, S_row[W - 1 + j][0]);
                scratchpad[j * blockDim.x + threadIdx.x] = nsj;
            }
            s0_reg[0] = pos1_reduce160_(cy, hi, lo);
        }
        scratchpad[threadIdx.x] = s0_reg;
    }

    // Second half: (HALF_F - 1) × (pow7+ARK fused + MVP(M)), then pow7 + MVP(M).
#pragma unroll 1
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        if constexpr (W == 16)
        {
#if POS1_MDS_CIRC
            pos1_round_fused_circ_<W>(&GPU_C_GL[(HALF_F + 1) * W + N_PART + r * W]);
#else
            pos1_round_fused_smem_<W>(&GPU_C_GL[(HALF_F + 1) * W + N_PART + r * W]);
#endif
        }
        else
        {
            pos1_pow7add_smem_<W>(&GPU_C_GL[(HALF_F + 1) * W + N_PART + r * W]);
            pos1_mvp_smem_mimm_<W>();
        }
    }
    if constexpr (W == 16)
    {
#if POS1_MDS_CIRC
        pos1_round_fused_circ_<W, false>();
#else
        pos1_round_fused_smem_<W, false>();
#endif
    }
    else
    {
        pos1_pow7_smem_<W>();
        pos1_mvp_smem_mimm_<W>();
    }

}

// ---------------------------------------------------------------------------
// Register-resident warp-cooperative permutation (one thread per state lane).
//
// A single sponge state is spread across lanes 0..W-1 of one warp: each lane
// keeps its state element in a register and pulls peer lanes' values with
// __shfl_sync. No shared memory, no __syncwarp (the shuffle's mask provides the
// ordering). `mask` must name exactly the active lanes (0..W-1). Entered only
// by active lanes; returns this lane's output.
// ---------------------------------------------------------------------------

// MVP: out_lane = sum_j mat[W*j + lane] * v_j, gathering v_j over the warp.
template<uint32_t W>
__device__ __forceinline__ gl64_t pos1_mvp_warp(gl64_t v, const gl64_t *mat, uint32_t lane, uint32_t mask)
{
    // 3-limb carry-chained accumulation, one canonical reduce160 per lane per
    // round -- replaces a fully reduced gl64 mul + add per term. Sound for any
    // full-width matrix entries (P as well as M).
    unsigned __int128 first = (unsigned __int128)mat[lane][0] * shfl_gl(mask, v, 0)[0];
    uint64_t lo = (uint64_t)first, hi = (uint64_t)(first >> 64);
    uint32_t cy = 0;
#pragma unroll
    for (uint32_t j = 1; j < W; ++j)
        pos1_mac_cc_(lo, hi, cy, mat[W * j + lane][0], shfl_gl(mask, v, j)[0]);
    gl64_t r; r[0] = pos1_reduce160_(cy, hi, lo);
    return r;
}

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__device__ gl64_t poseidon1PermuteWarpReg(gl64_t v,
                                          uint32_t lane,
                                          uint32_t mask,
                                          const gl64_t *GPU_C_GL,
                                          const gl64_t *GPU_S_GL,
                                          const gl64_t *GPU_M_GL,
                                          const gl64_t *GPU_P_GL)
{
    // ARK.
    v = v + GPU_C_GL[lane];

    // First half full rounds.
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        gl64_t x2 = v * v, x3 = v * x2, x4 = x2 * x2;
        v = (x3 * x4) + GPU_C_GL[(r + 1) * W + lane];
        v = pos1_mvp_warp<W>(v, GPU_M_GL, lane, mask);
    }

    // Transition full round → MVP(P).
    {
        gl64_t x2 = v * v, x3 = v * x2, x4 = x2 * x2;
        v = (x3 * x4) + GPU_C_GL[HALF_F * W + lane];
        v = pos1_mvp_warp<W>(v, GPU_P_GL, lane, mask);
    }

    // Partial rounds. Lane 0 owns the S-box; `a` (post-pow7 state[0]) drives the
    // sparse rank-1 update. The dot product s0 = Σ_j v_j·S_row[j] is formed as a
    // per-lane term then butterfly-reduced over the warp (log2(W) shuffles).
    for (uint32_t r = 0; r < N_PART; ++r)
    {
        if (lane == 0)
        {
            pow7(v);
            v = v + GPU_C_GL[(HALF_F + 1) * W + r];
        }
        gl64_t a = shfl_gl(mask, v, 0);   // post-pow7 state[0], broadcast to all

        const gl64_t *S_row = &GPU_S_GL[(W * 2 - 1) * r];
        gl64_t term = v * S_row[lane];    // this lane's contribution (lane 0: a·S_row[0])
        gl64_t s0 = warp_allreduce_add<W>(term, mask);

        v = (lane == 0) ? s0 : (v + a * S_row[W - 1 + lane]);
    }

    // Second half full rounds.
    for (uint32_t r = 0; r < HALF_F - 1; ++r)
    {
        gl64_t x2 = v * v, x3 = v * x2, x4 = x2 * x2;
        v = (x3 * x4) + GPU_C_GL[(HALF_F + 1) * W + N_PART + r * W + lane];
        v = pos1_mvp_warp<W>(v, GPU_M_GL, lane, mask);
    }
    // Final pow7 (no ARK) + MVP(M).
    {
        gl64_t x2 = v * v, x3 = v * x2, x4 = x2 * x2;
        v = x3 * x4;
        v = pos1_mvp_warp<W>(v, GPU_M_GL, lane, mask);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void permuteKernel_pos1(uint64_t *output, const uint64_t *input);

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART, uint32_t CAPACITY_T>
__global__ void permuteTruncKernel_pos1(uint64_t *output, const uint64_t *input);

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void linearHashKernel_pos1(uint64_t *__restrict__ output,
                                      uint64_t *__restrict__ input,
                                      uint32_t num_cols, uint32_t num_rows);

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART,
         uint32_t TPB_V, uint32_t MINB>
__global__ void __launch_bounds__(TPB_V, MINB) linearHashTiledKernel_pos1(uint64_t *__restrict__ output,
                                           uint64_t *__restrict__ input,
                                           uint32_t num_cols, uint32_t num_rows, Layout layout);

template<uint32_t RATE_T, uint32_t CAPACITY_T, uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void merkleNodeKernel_pos1(uint64_t nextN, uint64_t nextIndex,
                                      uint64_t pending, uint32_t arity,
                                      uint64_t *cursor);

template<uint32_t W, uint32_t HALF_F, uint32_t N_PART>
__global__ void grindingKernel_pos1(uint64_t *nonce, uint64_t *__restrict__ nonceBlock,
                                    uint64_t *__restrict__ input, uint64_t n_bits,
                                    uint64_t hashes_per_thread, uint64_t nonces_offset);

#endif  // POSEIDON_GOLDILOCKS_CUH
