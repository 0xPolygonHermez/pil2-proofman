#ifndef POSEIDON2_GOLDILOCKS
#define POSEIDON2_GOLDILOCKS

#include "poseidon2_goldilocks_constants.hpp"
#include "goldilocks_base_field.hpp"
#include <cstdio>
#include <cstdlib>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifndef HASH_SIZE
#define HASH_SIZE 4
#endif

#include "grinding_launch.hpp"  // NONCES_LAUNCH_* grinding launch geometry

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
    
    static void permuteTrunc(Goldilocks::Element (&state)[CAPACITY],
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
#ifdef __AVX2__
    inline void static add_avx(__m256i st[(SPONGE_WIDTH >> 2)], const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static pow7_avx(__m256i st[(SPONGE_WIDTH >> 2)]);
    inline void static add_avx_small(__m256i st[(SPONGE_WIDTH >> 2)], const Goldilocks::Element C[SPONGE_WIDTH]);
    inline void static matmul_external_avx(__m256i st[(SPONGE_WIDTH >> 2)]);
    inline void static matmul_external_batch_avx(__m256i *x);
    inline void static matmul_m4_batch_avx(__m256i &st0, __m256i &st1, __m256i &st2, __m256i &st3);
    inline void static pow7add_avx(__m256i *x, const Goldilocks::Element C_[SPONGE_WIDTH]);
    inline void static element_pow7_avx(__m256i &x);
#endif
#ifdef __AVX512__
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
    static void permuteTrunc_seq(Goldilocks::Element (&state)[CAPACITY],
                         const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    static void linear_hash_seq(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    static void merkletree_seq(Goldilocks::Element *tree, Goldilocks::Element *input,
                               uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                               int num_threads = 0, uint64_t dim = 1);

#ifdef __AVX2__
    // AVX2 single-sponge:
    static void permute_avx(Goldilocks::Element (&state)[SPONGE_WIDTH],
                            const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    static void permuteTrunc_avx(Goldilocks::Element (&state)[CAPACITY],
                         const Goldilocks::Element (&input)[SPONGE_WIDTH]);
    static void linear_hash_avx(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    static void merkletree_avx(Goldilocks::Element *tree, Goldilocks::Element *input,
                               uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                               int num_threads = 0, uint64_t dim = 1);
    // AVX2 4-lane batch (internal building blocks of merkletree_batch_avx):
    static void permute_batch_avx(Goldilocks::Element *, const Goldilocks::Element *);
    static void permuteTrunc_batch_avx(Goldilocks::Element (&state)[4 * CAPACITY],
                               const Goldilocks::Element (&input)[4 * SPONGE_WIDTH]);
    static void linear_hash_batch_avx(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    static void merkletree_batch_avx(Goldilocks::Element *tree, Goldilocks::Element *input,
                                     uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                                     int num_threads = 0, uint64_t dim = 1);
#endif
#ifdef __AVX512__
    // AVX512 8-lane batch (single-sponge AVX512 is intentionally not
    // implemented — see Poseidon2Mode enum comment).
    static void permute_batch_avx512(Goldilocks::Element *, const Goldilocks::Element *);
    static void permuteTrunc_batch_avx512(Goldilocks::Element (&state)[8 * CAPACITY],
                                  const Goldilocks::Element (&input)[8 * SPONGE_WIDTH]);
    static void linear_hash_batch_avx512(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    static void merkletree_batch_avx512(Goldilocks::Element *tree, Goldilocks::Element *input,
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
        Goldilocks::Element stored[4] = {Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero()};
        for (uint32_t i = 0; i < SPONGE_WIDTH; i+=4) {
            stored[0] = stored[0] + x[i];
            stored[1] = stored[1] + x[i+1];
            stored[2] = stored[2] + x[i+2];
            stored[3] = stored[3] + x[i+3];
        }
        
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i)
        {
            x[i] = x[i] + stored[i % 4];
        }
    }
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::permuteTrunc_seq(Goldilocks::Element (&state)[CAPACITY], Goldilocks::Element const (&input)[SPONGE_WIDTH])
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
    static const char *names[] = { "Auto", "Scalar", "Avx", "AvxBatch", "Avx512", "Avx512Batch" };
    int idx = static_cast<int>(m);
    const char *name = (idx >= 0 && idx < 6) ? names[idx] : "<unknown>";
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
#ifdef __AVX2__
        mode = Poseidon2Mode::Avx;
#else
        mode = Poseidon2Mode::Scalar;
#endif
    }
    switch (mode) {
        case Poseidon2Mode::Scalar: permute_seq(output, input); return;
#ifdef __AVX2__
        case Poseidon2Mode::Avx:    permute_avx(output, input); return;
#endif
        default: break;
    }
    abortMode("permute", mode);
}

template<uint32_t W>
inline void Poseidon2Goldilocks<W>::permuteTrunc(
    Goldilocks::Element (&state)[CAPACITY],
    const Goldilocks::Element (&input)[SPONGE_WIDTH],
    Poseidon2Mode mode)
{
    if (mode == Poseidon2Mode::Auto) {
#ifdef __AVX2__
        mode = Poseidon2Mode::Avx;
#else
        mode = Poseidon2Mode::Scalar;
#endif
    }
    switch (mode) {
        case Poseidon2Mode::Scalar: permuteTrunc_seq(state, input); return;
#ifdef __AVX2__
        case Poseidon2Mode::Avx:    permuteTrunc_avx(state, input); return;
#endif
        default: break;
    }
    abortMode("permuteTrunc", mode);
}

template<uint32_t W>
inline void Poseidon2Goldilocks<W>::linearHash(
    Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size, Poseidon2Mode mode)
{
    if (mode == Poseidon2Mode::Auto) {
#ifdef __AVX2__
        mode = Poseidon2Mode::Avx;
#else
        mode = Poseidon2Mode::Scalar;
#endif
    }
    switch (mode) {
        case Poseidon2Mode::Scalar: linear_hash_seq(output, input, size); return;
#ifdef __AVX2__
        case Poseidon2Mode::Avx:    linear_hash_avx(output, input, size); return;
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
#ifdef __AVX512__
        mode = Poseidon2Mode::Avx512Batch;
#elif defined(__AVX2__)
        mode = Poseidon2Mode::AvxBatch;
#else
        mode = Poseidon2Mode::Scalar;
#endif
    }
    switch (mode) {
        case Poseidon2Mode::Scalar:
            merkletree_seq(tree, input, num_cols, num_rows, arity, num_threads, dim); return;
#ifdef __AVX2__
        case Poseidon2Mode::Avx:
            merkletree_avx(tree, input, num_cols, num_rows, arity, num_threads, dim); return;
        case Poseidon2Mode::AvxBatch:
            merkletree_batch_avx(tree, input, num_cols, num_rows, arity, num_threads, dim); return;
#endif
#ifdef __AVX512__
        case Poseidon2Mode::Avx512Batch:
            merkletree_batch_avx512(tree, input, num_cols, num_rows, arity, num_threads, dim); return;
#endif
        // Avx512 single-sponge is intentionally unimplemented (see enum comment).
        default: break;
    }
    abortMode("merkletree", mode);
}

#include "poseidon2_goldilocks_avx.hpp"

#ifdef __AVX512__
 #include "poseidon2_goldilocks_avx512.hpp"
 #endif

using Poseidon2GoldilocksGrinding = Poseidon2Goldilocks<8>;  // SPONGE_WIDTH = 8

#endif