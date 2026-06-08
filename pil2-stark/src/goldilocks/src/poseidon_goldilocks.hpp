#ifndef POSEIDON_GOLDILOCKS
#define POSEIDON_GOLDILOCKS

#ifndef HASH_SIZE
#define HASH_SIZE 4
#endif

#include "poseidon_goldilocks_constants.hpp"
#include "goldilocks_base_field.hpp"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#ifdef __AVX2__
#include <immintrin.h>
#endif

// Mode selector for the Poseidon v1 public API.
// Auto resolves per-operation, inline, to the best backend compiled in.
// Explicit modes whose backend isn't compiled in abort loudly.
enum class PoseidonMode : uint8_t {
    Auto = 0,
    Scalar,
    Avx,
    AvxBatch,
    Avx512,
    Avx512Batch,
};


template<uint32_t SPONGE_WIDTH_T>
class PoseidonGoldilocks
{
public:

    static_assert(SPONGE_WIDTH_T == 8 || SPONGE_WIDTH_T == 12 || SPONGE_WIDTH_T == 16,
                  "SPONGE_WIDTH_T must be 8, 12, or 16");
    static constexpr uint32_t CAPACITY = 4;
    static constexpr uint32_t RATE = SPONGE_WIDTH_T - CAPACITY;
    static constexpr uint32_t SPONGE_WIDTH = SPONGE_WIDTH_T;
    static constexpr uint32_t N_FULL_ROUNDS_TOTAL = PoseidonGoldilocksConstants::ROUNDS_F;
    static constexpr uint32_t HALF_N_FULL_ROUNDS  = PoseidonGoldilocksConstants::ROUNDS_F_HALF;
    static constexpr uint32_t N_PARTIAL_ROUNDS    = PoseidonGoldilocksConstants::Poseidon1Tables<SPONGE_WIDTH_T>::N_PARTIAL_ROUNDS;
    static constexpr uint32_t N_ROUNDS = N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS;


    // Mode-dispatched public API (same shape as Poseidon2Goldilocks).

    static void permute(Goldilocks::Element (&output)[SPONGE_WIDTH],
                        const Goldilocks::Element (&input)[SPONGE_WIDTH],
                        PoseidonMode mode);

    static void permuteTrunc(Goldilocks::Element (&state)[CAPACITY],
                         const Goldilocks::Element (&input)[SPONGE_WIDTH],
                         PoseidonMode mode);

    static void linearHash(Goldilocks::Element *output, Goldilocks::Element *input,
                           uint64_t size, PoseidonMode mode);

    static void merkletree(Goldilocks::Element *tree, Goldilocks::Element *input,
                           uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                           PoseidonMode mode = PoseidonMode::Auto,
                           int num_threads = 0, uint64_t dim = 1);

    static void merkletreeReduce(Goldilocks::Element *root, Goldilocks::Element *input,
                                 uint64_t num_elements, uint64_t arity);

    static void grinding(uint64_t &out_idx, const uint64_t *in, const uint32_t n_bits);

private:
    inline static void pow7(Goldilocks::Element &x);
    inline static void pow7_(Goldilocks::Element *x);
    inline static void add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline static void pow7add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline static void mvp_(Goldilocks::Element *state, const Goldilocks::Element mat[SPONGE_WIDTH][SPONGE_WIDTH]);
    inline static Goldilocks::Element dot_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline static void prod_(Goldilocks::Element *x, const Goldilocks::Element alpha, const Goldilocks::Element C[SPONGE_WIDTH]);

#ifdef __AVX2__
    // 3-register single-sponge primitives (used by permute_avx).
    inline static void add_avx(__m256i &st0, __m256i &st1, __m256i &st2, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline static void pow7_avx(__m256i &st0, __m256i &st1, __m256i &st2);
    inline static void add_avx_a(__m256i &st0, __m256i &st1, __m256i &st2, const Goldilocks::Element C[SPONGE_WIDTH]);
    inline static void add_avx_small(__m256i &st0, __m256i &st1, __m256i &st2, const Goldilocks::Element C_small[SPONGE_WIDTH]);
#endif

    [[noreturn]] static void abortMode(const char *op, PoseidonMode m);

    // ---- Implementation primitives (reach via Mode parameter, never directly).

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
    // AVX2 4-lane batch:
    static void permute_batch_avx(Goldilocks::Element *, const Goldilocks::Element *);
    static void permuteTrunc_batch_avx(Goldilocks::Element (&state)[4 * CAPACITY],
                                   const Goldilocks::Element (&input)[4 * SPONGE_WIDTH]);
    static void linear_hash_batch_avx(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    static void merkletree_batch_avx(Goldilocks::Element *tree, Goldilocks::Element *input,
                                     uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                                     int num_threads = 0, uint64_t dim = 1);
#endif
#ifdef __AVX512__
    // AVX512 8-sponge batch: 12 __m512i registers, each holding one state
    // element across 8 sponges (strided layout).
    static void permute_batch_avx512(Goldilocks::Element *, const Goldilocks::Element *);
    static void permuteTrunc_batch_avx512(Goldilocks::Element (&state)[8 * CAPACITY],
                                      const Goldilocks::Element (&input)[8 * SPONGE_WIDTH]);
    static void linear_hash_batch_avx512(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size);
    static void merkletree_batch_avx512(Goldilocks::Element *tree, Goldilocks::Element *input,
                                        uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                                        int num_threads = 0, uint64_t dim = 1);
#endif
};

// ---------------------------------------------------------------------------
// Inline scalar primitives.
// ---------------------------------------------------------------------------

template<uint32_t W>
inline void PoseidonGoldilocks<W>::pow7(Goldilocks::Element &x)
{
    Goldilocks::Element x2 = x * x;
    Goldilocks::Element x3 = x * x2;
    Goldilocks::Element x4 = x2 * x2;
    x = x3 * x4;
}

template<uint32_t W>
inline void PoseidonGoldilocks<W>::pow7_(Goldilocks::Element *x)
{
    Goldilocks::Element x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i]  = x3[i] * x4[i];
    }
}

template<uint32_t W>
inline void PoseidonGoldilocks<W>::add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH])
{
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        x[i] = x[i] + C[i];
    }
}

template<uint32_t W>
inline void PoseidonGoldilocks<W>::prod_(Goldilocks::Element *x, const Goldilocks::Element alpha, const Goldilocks::Element C[SPONGE_WIDTH])
{
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        x[i] = alpha * C[i];
    }
}

template<uint32_t W>
inline void PoseidonGoldilocks<W>::pow7add_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH])
{
    Goldilocks::Element x2[SPONGE_WIDTH], x3[SPONGE_WIDTH], x4[SPONGE_WIDTH];
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        x2[i] = x[i] * x[i];
        x3[i] = x[i] * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i]  = x3[i] * x4[i];
        x[i]  = x[i] + C[i];
    }
}

template<uint32_t W>
inline Goldilocks::Element PoseidonGoldilocks<W>::dot_(Goldilocks::Element *x, const Goldilocks::Element C[SPONGE_WIDTH])
{
    Goldilocks::Element s0 = x[0] * C[0];
    for (uint32_t i = 1; i < SPONGE_WIDTH; ++i) {
        s0 = s0 + x[i] * C[i];
    }
    return s0;
}

template<uint32_t W>
inline void PoseidonGoldilocks<W>::mvp_(Goldilocks::Element *state, const Goldilocks::Element mat[SPONGE_WIDTH][SPONGE_WIDTH])
{
    // mat is applied transposed: indexed [j][i], not [i][j]
    Goldilocks::Element old_state[SPONGE_WIDTH];
    std::memcpy(old_state, state, sizeof(Goldilocks::Element) * SPONGE_WIDTH);
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        state[i] = mat[0][i] * old_state[0];
    }
    for (uint32_t j = 1; j < SPONGE_WIDTH; ++j) {
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
            state[i] = state[i] + (mat[j][i] * old_state[j]);
        }
    }
}

// ---------------------------------------------------------------------------
// permuteTrunc_seq lives inline so merkletree can call it from the header path.
// ---------------------------------------------------------------------------

template<uint32_t W>
inline void PoseidonGoldilocks<W>::permuteTrunc_seq(
    Goldilocks::Element (&state)[CAPACITY],
    const Goldilocks::Element (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    permute_seq(aux, input);
    for (uint32_t i = 0; i < CAPACITY; i++) state[i] = aux[i];
}

// ---------------------------------------------------------------------------
// Mode-dispatched API — inline template definitions.
// ---------------------------------------------------------------------------

template<uint32_t W>
[[noreturn]] inline void PoseidonGoldilocks<W>::abortMode(const char *op, PoseidonMode m)
{
    static const char *names[] = { "Auto", "Scalar", "Avx", "AvxBatch", "Avx512", "Avx512Batch" };
    int idx = static_cast<int>(m);
    const char *name = (idx >= 0 && idx < 6) ? names[idx] : "<unknown>";
    std::fprintf(stderr,
        "PoseidonGoldilocks<%u>::%s: mode %s is not available in this build "
        "(not compiled in, or not valid for this operation)\n",
        W, op, name);
    std::abort();
}

template<uint32_t W>
inline void PoseidonGoldilocks<W>::permute(
    Goldilocks::Element (&output)[SPONGE_WIDTH],
    const Goldilocks::Element (&input)[SPONGE_WIDTH],
    PoseidonMode mode)
{
    if (mode == PoseidonMode::Auto) {
#ifdef __AVX2__
        mode = (W == 12) ? PoseidonMode::Avx : PoseidonMode::Scalar;
#else
        mode = PoseidonMode::Scalar;
#endif
    }
    switch (mode) {
        case PoseidonMode::Scalar: permute_seq(output, input); return;
#ifdef __AVX2__
        case PoseidonMode::Avx:
            if constexpr (W == 12) { permute_avx(output, input); return; }
            break; 
#endif
        default: break;
    }
    abortMode("permute", mode);
}

template<uint32_t W>
inline void PoseidonGoldilocks<W>::permuteTrunc(
    Goldilocks::Element (&state)[CAPACITY],
    const Goldilocks::Element (&input)[SPONGE_WIDTH],
    PoseidonMode mode)
{
    if (mode == PoseidonMode::Auto) {
#ifdef __AVX2__
        mode = (W == 12) ? PoseidonMode::Avx : PoseidonMode::Scalar;
#else
        mode = PoseidonMode::Scalar;
#endif
    }
    switch (mode) {
        case PoseidonMode::Scalar: permuteTrunc_seq(state, input); return;
#ifdef __AVX2__
        case PoseidonMode::Avx:
            if constexpr (W == 12) { permuteTrunc_avx(state, input); return; }
            break;
#endif
        default: break;
    }
    abortMode("permuteTrunc", mode);
}

template<uint32_t W>
inline void PoseidonGoldilocks<W>::linearHash(
    Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size, PoseidonMode mode)
{
    if (mode == PoseidonMode::Auto) {
#ifdef __AVX2__
        mode = (W == 12) ? PoseidonMode::Avx : PoseidonMode::Scalar;
#else
        mode = PoseidonMode::Scalar;
#endif
    }
    switch (mode) {
        case PoseidonMode::Scalar: linear_hash_seq(output, input, size); return;
#ifdef __AVX2__
        case PoseidonMode::Avx:
            if constexpr (W == 12) { linear_hash_avx(output, input, size); return; }
            break;
#endif
        default: break;
    }
    abortMode("linearHash", mode);
}

template<uint32_t W>
inline void PoseidonGoldilocks<W>::merkletree(
    Goldilocks::Element *tree, Goldilocks::Element *input,
    uint64_t num_cols, uint64_t num_rows, uint64_t arity,
    PoseidonMode mode, int num_threads, uint64_t dim)
{
    if (mode == PoseidonMode::Auto) {
#ifdef __AVX512__
        mode = PoseidonMode::Avx512Batch;
#elif defined(__AVX2__)
        mode = PoseidonMode::AvxBatch;
#else
        mode = PoseidonMode::Scalar;
#endif
    }
    switch (mode) {
        case PoseidonMode::Scalar:
            merkletree_seq(tree, input, num_cols, num_rows, arity, num_threads, dim); return;
#ifdef __AVX2__
        case PoseidonMode::Avx:
            if constexpr (W == 12) {
                merkletree_avx(tree, input, num_cols, num_rows, arity, num_threads, dim); return;
            }
            break;
        case PoseidonMode::AvxBatch:
            merkletree_batch_avx(tree, input, num_cols, num_rows, arity, num_threads, dim); return;
#endif
#ifdef __AVX512__
        case PoseidonMode::Avx512Batch:
            merkletree_batch_avx512(tree, input, num_cols, num_rows, arity, num_threads, dim); return;
#endif
        default: break;
    }
    abortMode("merkletree", mode);
}

#include "poseidon_goldilocks_avx.hpp"

using PoseidonGoldilocksGrinding = PoseidonGoldilocks<8>;  // SPONGE_WIDTH = 8

#endif // POSEIDON_GOLDILOCKS

