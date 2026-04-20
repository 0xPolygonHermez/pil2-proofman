#ifndef POSEIDON2_GOLDILOCKS
#define POSEIDON2_GOLDILOCKS

#include "platform.hpp"
#include "poseidon2_goldilocks_constants.hpp"
#include "goldilocks_base_field.hpp"
#include <cstdio>
#include <cstdlib>
#if PIL2_HAS_AVX2
#include <immintrin.h>
#endif
#if PIL2_HAS_NEON
#include <arm_neon.h>
#endif

#define HASH_SIZE 4

// GPU PARAMS
#define NONCES_LAUNCH_BITS 19
#define NONCES_LAUNCH_BLOCKS 512
#define NONCES_LAUNCH_GRID_SIZE \
    (((1ULL << NONCES_LAUNCH_BITS) + NONCES_LAUNCH_BLOCKS - 1) / NONCES_LAUNCH_BLOCKS)

// Mode selector for the Poseidon2 public API.
// Auto resolves per-operation, inline, to the best backend compiled in.
// Explicit modes whose backend isn't compiled in abort loudly
enum class Poseidon2Mode : uint8_t {
    Auto = 0,
    Scalar,
    Avx,
    AvxBatch,
    Avx512,
    Avx512Batch,
    Neon,
    NeonBatch,
    ScalarUnrolledW16,
};


template<uint32_t SPONGE_WIDTH_T>
class Poseidon2Goldilocks
{
public:

    static_assert(SPONGE_WIDTH_T == 4 || SPONGE_WIDTH_T == 8 || SPONGE_WIDTH_T == 12 || SPONGE_WIDTH_T == 16, "SPONGE_WIDTH_T must be 4, 8, 12, or 16");
    static constexpr uint32_t RATE = SPONGE_WIDTH_T-4;
    static constexpr uint32_t CAPACITY = 4;
    static constexpr uint32_t SPONGE_WIDTH = SPONGE_WIDTH_T;
    static constexpr uint32_t N_FULL_ROUNDS_TOTAL = 8;
    static constexpr uint32_t HALF_N_FULL_ROUNDS = N_FULL_ROUNDS_TOTAL / 2;
    static constexpr uint32_t N_PARTIAL_ROUNDS = SPONGE_WIDTH_T == 4 ? 21 : 22;
    static constexpr uint32_t N_ROUNDS = N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS;


    // Mode-dispatched public API.

    static void permute(Goldilocks::Element (&output)[SPONGE_WIDTH],
                        const Goldilocks::Element (&input)[SPONGE_WIDTH],
                        Poseidon2Mode mode);
    
    static void compress(Goldilocks::Element (&state)[CAPACITY],
                     const Goldilocks::Element (&input)[SPONGE_WIDTH],
                     Poseidon2Mode mode);
    
    static void linearHash(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size, Poseidon2Mode mode);

    static void merkletree(Goldilocks::Element *tree, Goldilocks::Element *input,
                           uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                           Poseidon2Mode mode = Poseidon2Mode::Auto,
                           int num_threads = 0, uint64_t dim = 1);

    static void merkletreeReduce(Goldilocks::Element *root, Goldilocks::Element *input,
                                    uint64_t num_elements, uint64_t arity);

    static void grinding(uint64_t &out_idx, const uint64_t *in, const uint32_t n_bits);
   

private:
    inline void static pow7(Goldilocks::Element &x);
    inline void static pow7_(Goldilocks::Element *x);
    inline void static add_(Goldilocks::Element &x, const Goldilocks::Element *st);
    inline void static pow7add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static prodadd_(Goldilocks::Element *x, const Goldilocks::Element D[SPONGE_WIDTH], const Goldilocks::Element &sum);
    inline void static matmul_m4_(Goldilocks::Element *x);
    inline void static matmul_external_(Goldilocks::Element *x);
#if PIL2_HAS_AVX2
    inline void static add_avx(__m256i st[(SPONGE_WIDTH >> 2)], const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static pow7_avx(__m256i st[(SPONGE_WIDTH >> 2)]);
    inline void static add_avx_small(__m256i st[(SPONGE_WIDTH >> 2)], const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static matmul_external_avx(__m256i st[(SPONGE_WIDTH >> 2)]);
    inline void static matmul_external_batch_avx(__m256i *x);
    inline void static matmul_m4_batch_avx(__m256i &st0, __m256i &st1, __m256i &st2, __m256i &st3);
    inline void static pow7add_avx(__m256i *x, const Goldilocks::Element C_[SPONGE_WIDTH]);
    inline void static element_pow7_avx(__m256i &x);
#endif
#if PIL2_HAS_AVX512
    inline void static matmul_external_batch_avx512(__m512i *x);
    inline void static matmul_m4_batch_avx512(__m512i &st0, __m512i &st1, __m512i &st2, __m512i &st3);
    inline void static pow7add_avx512(__m512i *x, const Goldilocks::Element C_[SPONGE_WIDTH]);
    inline void static element_pow7_avx512(__m512i &x);
#endif

    [[noreturn]] static void abortMode(const char *op, Poseidon2Mode m);

    // ---- Implementation primitives (private — reach via Mode parameter).
    // These back the Mode API above. Reach them via Mode parameter, never directly.

    // Scalar:
    static void permute_seq(Goldilocks::Element (&state)[SPONGE_WIDTH],
                            const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    static void compress_seq(Goldilocks::Element (&state)[CAPACITY],
                         const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    static void linear_hash_seq(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    static void merkletree_seq(Goldilocks::Element *tree, Goldilocks::Element *input,
                               uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                               int num_threads = 0, uint64_t dim = 1);

#if PIL2_HAS_AVX2
    // AVX2 single-sponge:
    static void permute_avx(Goldilocks::Element (&state)[SPONGE_WIDTH],
                            const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    static void compress_avx(Goldilocks::Element (&state)[CAPACITY],
                         const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    static void linear_hash_avx(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    static void merkletree_avx(Goldilocks::Element *tree, Goldilocks::Element *input,
                               uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                               int num_threads = 0, uint64_t dim = 1);
    // AVX2 4-lane batch (internal building blocks of merkletree_batch_avx):
    static void permute_batch_avx(Goldilocks::Element *, const Goldilocks::Element *);
    static void compress_batch_avx(Goldilocks::Element (&state)[4 * CAPACITY],
                               const Goldilocks::Element (&input)[4 * SPONGE_WIDTH]);
    static void linear_hash_batch_avx(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    static void merkletree_batch_avx(Goldilocks::Element *tree, Goldilocks::Element *input,
                                     uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                                     int num_threads = 0, uint64_t dim = 1);
#endif
#if PIL2_HAS_AVX512
    // AVX512 8-lane batch (single-sponge AVX512 is intentionally not
    // implemented — see Poseidon2Mode enum comment).
    static void permute_batch_avx512(Goldilocks::Element *, const Goldilocks::Element *);
    static void compress_batch_avx512(Goldilocks::Element (&state)[8 * CAPACITY],
                                  const Goldilocks::Element (&input)[8 * SPONGE_WIDTH]);
    static void linear_hash_batch_avx512(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    static void merkletree_batch_avx512(Goldilocks::Element *tree, Goldilocks::Element *input,
                                        uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                                        int num_threads = 0, uint64_t dim = 1);
#endif
#if PIL2_HAS_NEON
    // NEON single-sponge (Part 5). 2 elements per uint64x2_t — state of W
    // elements lives in W/2 NEON registers. Bodies in poseidon2_goldilocks_neon.hpp.
    static void permute_neon(Goldilocks::Element *, const Goldilocks::Element *);
    static void compress_neon(Goldilocks::Element (&state)[CAPACITY],
                          const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    static void linear_hash_neon(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    static void merkletree_neon(Goldilocks::Element *tree, Goldilocks::Element *input,
                                uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                                int num_threads = 0, uint64_t dim = 1);
    // NEON 2-sponge batch (Part 5 Task 36). Each uint64x2_t holds the same
    // element index from 2 sponges. Mirrors AvxBatch (4-sponge) shape but
    // with NEON's 2-lane width. Bodies in poseidon2_goldilocks_neon.hpp.
    static void permute_batch_neon(Goldilocks::Element *, const Goldilocks::Element *);
    static void compress_batch_neon(Goldilocks::Element (&state)[2 * CAPACITY],
                                const Goldilocks::Element (&input)[2 * SPONGE_WIDTH]);
    static void linear_hash_batch_neon(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    static void merkletree_batch_neon(Goldilocks::Element *tree, Goldilocks::Element *input,
                                      uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                                      int num_threads = 0, uint64_t dim = 1);
    static void permute_w16_scalar_unrolled(Goldilocks::Element (&state)[SPONGE_WIDTH],
                                           const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    static void compress_w16_scalar_unrolled(Goldilocks::Element (&state)[CAPACITY],
                                            const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    static void linear_hash_w16_scalar_unrolled(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    static void merkletree_w16_scalar_unrolled(Goldilocks::Element *tree, Goldilocks::Element *input,
                                              uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                                              int num_threads = 0, uint64_t dim = 1);
#endif

};

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::pow7(Goldilocks::Element &x)
{
    Goldilocks::Element x2 = x * x;
    Goldilocks::Element x3 = x * x2;
    Goldilocks::Element x4 = x2 * x2;
    x = x3 * x4;
};

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::pow7_(Goldilocks::Element *x)
{
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i)
    {
        Goldilocks::Element x2 = x[i] * x[i];
        Goldilocks::Element x3 = x[i] * x2;
        Goldilocks::Element x4 = x2 * x2;
        x[i] = x3 * x4;
    }
};

    
template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::add_(Goldilocks::Element &x, const Goldilocks::Element *st)
{
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i)
    {
        x = x + st[i];
    }
}
template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::prodadd_(Goldilocks::Element *x, const Goldilocks::Element D[SPONGE_WIDTH], const Goldilocks::Element &sum)
{
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i)
    {
        x[i] = x[i]*D[i] + sum;
    }
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::pow7add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH])
{    
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i)
    {
        
        Goldilocks::Element xi = x[i] + C[i];
        Goldilocks::Element x2 = xi * xi;
        Goldilocks::Element x3 = xi * x2;
        Goldilocks::Element x4 = x2 * x2;
        x[i] = x3 * x4;
    }
};

    
template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::matmul_m4_(Goldilocks::Element *x) {
    Goldilocks::Element t0 = x[0] + x[1];
    Goldilocks::Element t1 = x[2] + x[3];
    Goldilocks::Element t2 = x[1] + x[1] + t1;
    Goldilocks::Element t3 = x[3] + x[3] + t0;
    Goldilocks::Element t1_2 = t1 + t1;
    Goldilocks::Element t0_2 = t0 + t0;
    Goldilocks::Element t4 = t1_2 + t1_2 + t3;
    Goldilocks::Element t5 = t0_2 + t0_2 + t2;
    Goldilocks::Element t6 = t3 + t5;
    Goldilocks::Element t7 = t2 + t4;
    
    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::matmul_external_(Goldilocks::Element *x) {

    for(uint32_t i = 0; i < SPONGE_WIDTH; i +=4) {
        matmul_m4_(&x[i]);
    }
    if(SPONGE_WIDTH > 4){
        // Pairwise reduction for the per-column accumulators. Same result as
        // the linear chain `stored[c] += x[i]; stored[c] += x[i+4]; ...` but
        // with chain depth log2(W/4) instead of W/4. For W=16 that's depth 2
        // vs 4; for W=12 depth 2 vs 3. Shorter critical path lets the CPU's
        // out-of-order engine issue the four per-column chains more densely,
        // and helps the NEON path too (its matmul_external_neon punts to
        // this scalar routine — see poseidon2_goldilocks_neon.hpp:69).
        // Ported from the Metal kernel optimisation at commit 6901b051.
        Goldilocks::Element stored[4];
        if constexpr (SPONGE_WIDTH == 16) {
            Goldilocks::Element p00 = x[0]  + x[4];
            Goldilocks::Element p01 = x[8]  + x[12];
            Goldilocks::Element p10 = x[1]  + x[5];
            Goldilocks::Element p11 = x[9]  + x[13];
            Goldilocks::Element p20 = x[2]  + x[6];
            Goldilocks::Element p21 = x[10] + x[14];
            Goldilocks::Element p30 = x[3]  + x[7];
            Goldilocks::Element p31 = x[11] + x[15];
            stored[0] = p00 + p01;
            stored[1] = p10 + p11;
            stored[2] = p20 + p21;
            stored[3] = p30 + p31;
        } else if constexpr (SPONGE_WIDTH == 12) {
            stored[0] = (x[0] + x[4]) + x[8];
            stored[1] = (x[1] + x[5]) + x[9];
            stored[2] = (x[2] + x[6]) + x[10];
            stored[3] = (x[3] + x[7]) + x[11];
        } else {  // SPONGE_WIDTH == 8 (already depth-1; keep straight add)
            stored[0] = x[0] + x[4];
            stored[1] = x[1] + x[5];
            stored[2] = x[2] + x[6];
            stored[3] = x[3] + x[7];
        }

        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i)
        {
            x[i] = x[i] + stored[i % 4];
        }
    }
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::compress_seq(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    permute_seq(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}

// ---------------------------------------------------------------------------
// Mode-dispatched API — inline template definitions.
// ---------------------------------------------------------------------------

template<uint32_t W>
[[noreturn]] inline void Poseidon2Goldilocks<W>::abortMode(const char *op, Poseidon2Mode m)
{
    static const char *names[] = { "Auto", "Scalar", "Avx", "AvxBatch", "Avx512", "Avx512Batch", "Neon", "NeonBatch", "ScalarUnrolledW16" };
    int idx = static_cast<int>(m);
    const char *name = (idx >= 0 && idx < (int)(sizeof(names) / sizeof(*names))) ? names[idx] : "<unknown>";
    std::fprintf(stderr,
        "Poseidon2Goldilocks<%u>::%s: mode %s is not available in this build "
        "(not compiled in, or not valid for this operation)\n",
        W, op, name);
    std::abort();
}

template<uint32_t W>
inline void Poseidon2Goldilocks<W>::permute(
    Goldilocks::Element (&output)[SPONGE_WIDTH],
    const Goldilocks::Element (&input)[SPONGE_WIDTH],
    Poseidon2Mode mode)
{
    if (mode == Poseidon2Mode::Auto) {
#if PIL2_HAS_AVX2
        mode = Poseidon2Mode::Avx;
#elif PIL2_HAS_NEON
        // NEON wins at W=4 (~9%) and W=8 (~5%) but regresses at W=12/W=16
        // because matmul_external_neon punts to scalar and the per-call
        // NEON-store / scalar / NEON-load overhead scales with W. Restrict
        // Auto to W=4/W=8 for the generic NEON path. W=16 uses a separate
        // scalar-shaped NEON candidate that avoids that overhead. Explicit
        // Mode::Neon still works for any W (correctness gated).
        mode = (W == 16) ? Poseidon2Mode::ScalarUnrolledW16 : ((W == 4 || W == 8) ? Poseidon2Mode::Neon : Poseidon2Mode::Scalar);
#else
        mode = Poseidon2Mode::Scalar;
#endif
    }
    switch (mode) {
        case Poseidon2Mode::Scalar: permute_seq(output, input); return;
#if PIL2_HAS_AVX2
        case Poseidon2Mode::Avx:    permute_avx(output, input); return;
#endif
#if PIL2_HAS_NEON
        case Poseidon2Mode::Neon:
            // NEON wins only at W=4/W=8; W=12/W=16 dispatch is a compile-
            // time fall-through that aborts at runtime (Auto never picks it).
            if constexpr (W == 4 || W == 8) {
                permute_neon(output, input);
                return;
            }
            break;
        case Poseidon2Mode::ScalarUnrolledW16:
            if constexpr (W == 16) {
                permute_w16_scalar_unrolled(output, input);
                return;
            }
            break;
#endif
        default: break;
    }
    abortMode("permute", mode);
}

template<uint32_t W>
inline void Poseidon2Goldilocks<W>::compress(
    Goldilocks::Element (&state)[CAPACITY],
    const Goldilocks::Element (&input)[SPONGE_WIDTH],
    Poseidon2Mode mode)
{
    if (mode == Poseidon2Mode::Auto) {
#if PIL2_HAS_AVX2
        mode = Poseidon2Mode::Avx;
#elif PIL2_HAS_NEON
        // See permute() Auto comment.
        mode = (W == 16) ? Poseidon2Mode::ScalarUnrolledW16 : ((W == 4 || W == 8) ? Poseidon2Mode::Neon : Poseidon2Mode::Scalar);
#else
        mode = Poseidon2Mode::Scalar;
#endif
    }
    switch (mode) {
        case Poseidon2Mode::Scalar: compress_seq(state, input); return;
#if PIL2_HAS_AVX2
        case Poseidon2Mode::Avx:    compress_avx(state, input); return;
#endif
#if PIL2_HAS_NEON
        case Poseidon2Mode::Neon:
            if constexpr (W == 4 || W == 8) {
                compress_neon(state, input);
                return;
            }
            break;
        case Poseidon2Mode::ScalarUnrolledW16:
            if constexpr (W == 16) {
                compress_w16_scalar_unrolled(state, input);
                return;
            }
            break;
#endif
        default: break;
    }
    abortMode("compress", mode);
}

template<uint32_t W>
inline void Poseidon2Goldilocks<W>::linearHash(
    Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size, Poseidon2Mode mode)
{
    if (mode == Poseidon2Mode::Auto) {
#if PIL2_HAS_AVX2
        mode = Poseidon2Mode::Avx;
#elif PIL2_HAS_NEON
        // See permute() Auto comment.
        mode = (W == 16) ? Poseidon2Mode::ScalarUnrolledW16 : ((W == 4 || W == 8) ? Poseidon2Mode::Neon : Poseidon2Mode::Scalar);
#else
        mode = Poseidon2Mode::Scalar;
#endif
    }
    switch (mode) {
        case Poseidon2Mode::Scalar: linear_hash_seq(output, input, size); return;
#if PIL2_HAS_AVX2
        case Poseidon2Mode::Avx:    linear_hash_avx(output, input, size); return;
#endif
#if PIL2_HAS_NEON
        case Poseidon2Mode::Neon:
            if constexpr (W == 4 || W == 8) {
                linear_hash_neon(output, input, size);
                return;
            }
            break;
        case Poseidon2Mode::ScalarUnrolledW16:
            if constexpr (W == 16) {
                linear_hash_w16_scalar_unrolled(output, input, size);
                return;
            }
            break;
#endif
        // AvxBatch / Avx512Batch have a 4/8-row contract and are not callable
        // as single-row linearHash — they remain private, reachable only via
        // merkletree().
        default: break;
    }
    abortMode("linearHash", mode);
}

template<uint32_t W>
inline void Poseidon2Goldilocks<W>::merkletree(
    Goldilocks::Element *tree, Goldilocks::Element *input,
    uint64_t num_cols, uint64_t num_rows, uint64_t arity,
    Poseidon2Mode mode, int num_threads, uint64_t dim)
{
    if (mode == Poseidon2Mode::Auto) {
#if PIL2_HAS_AVX512
        mode = Poseidon2Mode::Avx512Batch;
#elif PIL2_HAS_AVX2
        mode = Poseidon2Mode::AvxBatch;
#elif PIL2_HAS_NEON
        // NeonBatch is correctness-gated for all widths but is a 5-16% perf
        // regression vs Scalar on M4 Pro at the merkletree level. Strided
        // gather loads + clang's auto-vectorisation of the scalar path leave
        // NEON's 2-lane parallelism unable to compensate. Keep batch impls
        // available via explicit Mode::NeonBatch (tests verify), but Auto
        // stays on the path that actually wins. W=16 uses the scalar-shaped
        // NEON candidate because the reference proving key's arity-4 Merkle
        // path benefits from it. Other widths stay Scalar at the tree level.
        mode = (W == 16) ? Poseidon2Mode::ScalarUnrolledW16 : Poseidon2Mode::Scalar;
#else
        mode = Poseidon2Mode::Scalar;
#endif
    }
    switch (mode) {
        case Poseidon2Mode::Scalar:
            merkletree_seq(tree, input, num_cols, num_rows, arity, num_threads, dim); return;
#if PIL2_HAS_AVX2
        case Poseidon2Mode::Avx:
            merkletree_avx(tree, input, num_cols, num_rows, arity, num_threads, dim); return;
        case Poseidon2Mode::AvxBatch:
            merkletree_batch_avx(tree, input, num_cols, num_rows, arity, num_threads, dim); return;
#endif
#if PIL2_HAS_AVX512
        case Poseidon2Mode::Avx512Batch:
            merkletree_batch_avx512(tree, input, num_cols, num_rows, arity, num_threads, dim); return;
#endif
#if PIL2_HAS_NEON
        case Poseidon2Mode::Neon:
            if constexpr (W == 4 || W == 8) {
                merkletree_neon(tree, input, num_cols, num_rows, arity, num_threads, dim);
                return;
            }
            break;
        case Poseidon2Mode::NeonBatch:
            if constexpr (W == 4 || W == 8) {
                merkletree_batch_neon(tree, input, num_cols, num_rows, arity, num_threads, dim);
                return;
            }
            break;
        case Poseidon2Mode::ScalarUnrolledW16:
            if constexpr (W == 16) {
                merkletree_w16_scalar_unrolled(tree, input, num_cols, num_rows, arity, num_threads, dim);
                return;
            }
            break;
#endif
        // Avx512 single-sponge is intentionally unimplemented (see enum comment).
        default: break;
    }
    abortMode("merkletree", mode);
}

#include "poseidon2_goldilocks_avx.hpp"

#if PIL2_HAS_AVX512
 #include "poseidon2_goldilocks_avx512.hpp"
 #endif

#include "poseidon2_goldilocks_neon.hpp"

using Poseidon2GoldilocksGrinding = Poseidon2Goldilocks<4>;  // SPONGE_WIDTH = 4

#endif
