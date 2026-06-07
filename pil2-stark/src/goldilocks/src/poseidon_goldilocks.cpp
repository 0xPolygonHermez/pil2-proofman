#include "poseidon_goldilocks.hpp"
#include <math.h> /* floor */
#include <omp.h>
#include <cassert>
#include <atomic>
#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Scalar
// ---------------------------------------------------------------------------

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::permute_seq(
    Goldilocks::Element (&state)[SPONGE_WIDTH],
    const Goldilocks::Element (&input)[SPONGE_WIDTH])
{
    using Tbl = PoseidonGoldilocksConstants::Poseidon1Tables<SPONGE_WIDTH_T>;
    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);

    std::memcpy(state, input, length);

    add_(state, &(Tbl::C[0]));
    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7add_(state, &(Tbl::C[(r + 1) * SPONGE_WIDTH]));
        mvp_(state, Tbl::M);
    }
    pow7add_(state, &(Tbl::C[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
    mvp_(state, Tbl::P);

    for (uint32_t r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        pow7(state[0]);
        state[0] = state[0] + Tbl::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        Goldilocks::Element s0 = dot_(state, &(Tbl::S[(SPONGE_WIDTH * 2 - 1) * r]));
        Goldilocks::Element W_[SPONGE_WIDTH];
        prod_(W_, state[0], &(Tbl::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1]));
        add_(state, W_);
        state[0] = s0;
    }

    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7add_(state, &(Tbl::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        mvp_(state, Tbl::M);
    }
    pow7_(&(state[0]));
    mvp_(state, Tbl::M);

}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::linear_hash_seq(
    Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[SPONGE_WIDTH];

    while (remaining)
    {
        if (remaining == size)
        {
            memset(state + RATE, 0, CAPACITY * sizeof(Goldilocks::Element));
        }
        else
        {
// avoid -Wrestrict warning, there is no overlapping in practice
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
            std::memcpy(state + RATE, state, CAPACITY * sizeof(Goldilocks::Element));
#pragma GCC diagnostic pop
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        memset(&state[n], 0, (RATE - n) * sizeof(Goldilocks::Element));
        std::memcpy(state, input + (size - remaining), n * sizeof(Goldilocks::Element));
        permute_seq(state, state);
        remaining -= n;
    }
    if (size > 0)
    {
        std::memcpy(output, state, CAPACITY * sizeof(Goldilocks::Element));
    }
    else
    {
        memset(output, 0, CAPACITY * sizeof(Goldilocks::Element));
    }
}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::merkletree_seq(
    Goldilocks::Element *tree, Goldilocks::Element *input,
    uint64_t num_cols, uint64_t num_rows, uint64_t arity,
    int num_threads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }

    Goldilocks::Element *cursor = tree;
    if (num_threads == 0)
        num_threads = omp_get_max_threads();

#pragma omp parallel for num_threads(num_threads)
    for (uint64_t i = 0; i < num_rows; i++)
    {
        linear_hash_seq(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
    }

    // Build the merkle tree (arity-generic reduction).
    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
        {
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
        }

    #pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));

            // Read arity*CAPACITY child hashes (CAPACITY elements each) and
            // zero-pad the remaining SPONGE_WIDTH slots.
            std::memcpy(pol_input, &cursor[nextIndex + i * (arity * CAPACITY)], (arity * CAPACITY) * sizeof(Goldilocks::Element));

            permuteTrunc_seq((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::merkletreeReduce(
    Goldilocks::Element *root, Goldilocks::Element *input,
    uint64_t num_elements, uint64_t arity)
{
    assert(arity * CAPACITY <= SPONGE_WIDTH &&
           "PoseidonGoldilocks::merkletreeReduce: arity*CAPACITY exceeds SPONGE_WIDTH");
    uint64_t numNodes = num_elements;
    uint64_t nodesLevel = num_elements;

    while (nodesLevel > 1) {
        uint64_t extraZeros = (arity - (nodesLevel % arity)) % arity;
        numNodes += extraZeros;
        uint64_t nextN = (nodesLevel + (arity - 1)) / arity;
        numNodes += nextN;
        nodesLevel = nextN;
    }

    Goldilocks::Element *cursor = new Goldilocks::Element[numNodes * CAPACITY];
    memcpy(cursor, input, num_elements * CAPACITY * sizeof(Goldilocks::Element));

    // Build the merkle tree
    uint64_t pending = num_elements;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
        {
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
        }

        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));

            // Read arity*CAPACITY child hashes (CAPACITY elements each) and
            // zero-pad the remaining SPONGE_WIDTH slots.
            std::memcpy(pol_input, &cursor[nextIndex + i * (arity * CAPACITY)], (arity * CAPACITY) * sizeof(Goldilocks::Element));

            permuteTrunc_seq((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }

    std::memcpy(root, &cursor[nextIndex], CAPACITY * sizeof(Goldilocks::Element));
    delete[] cursor;
}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::grinding(
    uint64_t &nonce, const uint64_t *in, const uint32_t n_bits)
{
    // Total hashes needed for S-bit security at n_bits target difficulty:
    //   N = -S · ln(2) / ln(1 - 2^-n_bits).  Same formula as the GPU path;
    // log1p/ldexp keep precision for small 2^-n_bits.
    constexpr uint64_t security = 128;
    const uint64_t level = uint64_t(1) << (64 - n_bits);
    const double eps   = std::ldexp(1.0, -int(n_bits));
    const double total = -double(security) * std::log(2.0) / std::log1p(-eps);
    const uint64_t N   = (uint64_t)std::ceil(total);

    std::atomic<uint64_t> found{UINT64_MAX};

    #pragma omp parallel
    {
        const int tid  = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
       
        constexpr uint64_t POLL_MASK = 15;
        uint64_t local_found = UINT64_MAX;
        uint64_t poll = 0;

        // STARK grinding contract:
        //   state[0..2] = FIELD_EXTENSION challenge (in[0..2])
        //   state[3]    = nonce (set per iteration below)
        //   state[4..W-1] = 0 (zero-padding)
        Goldilocks::Element state[SPONGE_WIDTH] = {};
        Goldilocks::Element out[SPONGE_WIDTH];
        std::memcpy(state, in, 3 * sizeof(Goldilocks::Element));

        for (uint64_t i = tid; i < N; i += nthr)
        {
            if ((poll++ & POLL_MASK) == 0)
                local_found = found.load(std::memory_order_relaxed);
            if (i >= local_found) break;

            state[3] = Goldilocks::fromU64(i);
            permute_seq(out, state);
            if (out[0].fe < level) {
                #pragma omp critical(grinding_update)
                {
                    if (i < found.load(std::memory_order_relaxed))
                        found.store(i, std::memory_order_relaxed);
                }
                break;
            }
        }
    }

    nonce = found.load();
    if (nonce == UINT64_MAX)
    {
        throw std::runtime_error("PoseidonGoldilocks::grinding: could not find a valid nonce");
    }
}

// ---------------------------------------------------------------------------
// AVX2
// ---------------------------------------------------------------------------

#ifdef __AVX2__

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::permute_avx(
    Goldilocks::Element (&state)[SPONGE_WIDTH],
    const Goldilocks::Element (&input)[SPONGE_WIDTH])
{
if constexpr (SPONGE_WIDTH_T != 12) { abortMode("permute_avx", PoseidonMode::Avx); } else {
    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);

    std::memcpy(state, input, length);
    __m256i st0, st1, st2;
    Goldilocks::load_avx(st0, &(state[0]));
    Goldilocks::load_avx(st1, &(state[4]));
    Goldilocks::load_avx(st2, &(state[8]));
    add_avx_small(st0, st1, st2, &(PoseidonGoldilocksConstants::C12[0]));

    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_avx(st0, st1, st2);
        add_avx_small(st0, st1, st2, &(PoseidonGoldilocksConstants::C12[(r + 1) * SPONGE_WIDTH]));
        Goldilocks::mmult_avx_8(st0, st1, st2, &(PoseidonGoldilocksConstants::M_12[0]));
    }
    pow7_avx(st0, st1, st2);
    add_avx(st0, st1, st2, &(PoseidonGoldilocksConstants::C12[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
    Goldilocks::mmult_avx(st0, st1, st2, &(PoseidonGoldilocksConstants::P_12[0]));

    Goldilocks::store_avx(&(state[0]), st0);
    Goldilocks::Element state0_ = state[0];
    Goldilocks::Element state0;

    __m256i mask = _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0);
    for (uint32_t r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        state0 = state0_;
        pow7(state0);
        state0 = state0 + PoseidonGoldilocksConstants::C12[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        state0_ = state0 * PoseidonGoldilocksConstants::S12[(SPONGE_WIDTH * 2 - 1) * r];
        st0 = _mm256_and_si256(st0, mask);
        state0_ = state0_ + Goldilocks::dot_avx(st0, st1, st2, &(PoseidonGoldilocksConstants::S12[(SPONGE_WIDTH * 2 - 1) * r]));
        __m256i scalar1 = _mm256_set1_epi64x(state0.fe);
        __m256i w0, w1, w2, s0, s1, s2;
        Goldilocks::load_avx(s0, &(PoseidonGoldilocksConstants::S12[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1]));
        Goldilocks::load_avx(s1, &(PoseidonGoldilocksConstants::S12[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1 + 4]));
        Goldilocks::load_avx(s2, &(PoseidonGoldilocksConstants::S12[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1 + 8]));
        Goldilocks::mult_avx(w0, scalar1, s0);
        Goldilocks::mult_avx(w1, scalar1, s1);
        Goldilocks::mult_avx(w2, scalar1, s2);
        Goldilocks::add_avx(st0, st0, w0);
        Goldilocks::add_avx(st1, st1, w1);
        Goldilocks::add_avx(st2, st2, w2);
        state0 = state0 + PoseidonGoldilocksConstants::S12[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1];
    }
    Goldilocks::store_avx(&(state[0]), st0);
    state[0] = state0_;
    Goldilocks::load_avx(st0, &(state[0]));

    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7_avx(st0, st1, st2);
        add_avx_small(st0, st1, st2, &(PoseidonGoldilocksConstants::C12[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        Goldilocks::mmult_avx_8(st0, st1, st2, &(PoseidonGoldilocksConstants::M_12[0]));
    }
    pow7_avx(st0, st1, st2);
    Goldilocks::mmult_avx_8(st0, st1, st2, &(PoseidonGoldilocksConstants::M_12[0]));

    Goldilocks::store_avx(&(state[0]), st0);
    Goldilocks::store_avx(&(state[4]), st1);
    Goldilocks::store_avx(&(state[8]), st2);

}
}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::permuteTrunc_avx(
    Goldilocks::Element (&state)[CAPACITY],
    const Goldilocks::Element (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    permute_avx(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::linear_hash_avx(
    Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[SPONGE_WIDTH];

    while (remaining)
    {
        if (remaining == size)
        {
            memset(state + RATE, 0, CAPACITY * sizeof(Goldilocks::Element));
        }
        else
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
            std::memcpy(state + RATE, state, CAPACITY * sizeof(Goldilocks::Element));
#pragma GCC diagnostic pop
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        memset(&state[n], 0, (RATE - n) * sizeof(Goldilocks::Element));
        std::memcpy(state, input + (size - remaining), n * sizeof(Goldilocks::Element));
        permute_avx(state, state);
        remaining -= n;
    }
    if (size > 0)
    {
        std::memcpy(output, state, CAPACITY * sizeof(Goldilocks::Element));
    }
    else
    {
        memset(output, 0, CAPACITY * sizeof(Goldilocks::Element));
    }
}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::merkletree_avx(
    Goldilocks::Element *tree, Goldilocks::Element *input,
    uint64_t num_cols, uint64_t num_rows, uint64_t arity,
    int num_threads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    Goldilocks::Element *cursor = tree;
    if (num_threads == 0)
        num_threads = omp_get_max_threads();

#pragma omp parallel for num_threads(num_threads)
    for (uint64_t i = 0; i < num_rows; i++)
    {
        linear_hash_avx(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
    }

    // Build the merkle tree (arity-generic).
    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
        {
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
        }

    #pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));

            // Read arity*CAPACITY child hashes (CAPACITY elements each) and
            // zero-pad the remaining SPONGE_WIDTH slots.
            std::memcpy(pol_input, &cursor[nextIndex + i * (arity * CAPACITY)], (arity * CAPACITY) * sizeof(Goldilocks::Element));

            permuteTrunc_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

// --- AVX2 4-sponge batch ---------------------------------------------------
// Each of the 12 __m256i registers holds one state element across 4 sponges
// (strided layout). Full-round matmul uses M12[j][i] indexing, the transition
// uses P12, and the partial round applies the S-matrix rank-1 update.

namespace {
#ifdef __AVX2__
inline void element_pow7_avx(__m256i &x) {
    __m256i x2, x3, x4;
    Goldilocks::square_avx(x2, x);
    Goldilocks::mult_avx(x3, x, x2);
    Goldilocks::square_avx(x4, x2);
    Goldilocks::mult_avx(x, x3, x4);
}
#endif
#ifdef __AVX512__
inline void element_pow7_avx512(__m512i &x) {
    __m512i x2, x3, x4;
    Goldilocks::square_avx512(x2, x);
    Goldilocks::mult_avx512(x3, x, x2);
    Goldilocks::square_avx512(x4, x2);
    Goldilocks::mult_avx512(x, x3, x4);
}
#endif
} // namespace

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::permute_batch_avx(
    Goldilocks::Element *state, const Goldilocks::Element *input)
{

    std::memcpy(state, input, 4 * SPONGE_WIDTH * sizeof(Goldilocks::Element));

    using Tbl = PoseidonGoldilocksConstants::Poseidon1Tables<SPONGE_WIDTH_T>;

    __m256i st[SPONGE_WIDTH];
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        Goldilocks::load_avx(st[i], &state[i], SPONGE_WIDTH);
    }

    // Initial constant add.
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        __m256i c_bc = _mm256_set1_epi64x(Tbl::C[i].fe);
        Goldilocks::add_avx(st[i], st[i], c_bc);
    }

    auto matmul_M = [&]() {
        // M12 entries fit in 8 bits → use 72-bit multiplier and accumulate
        // products pre-reduction (12× fewer reductions vs per-mult reduce).
        __m256i new_st[SPONGE_WIDTH];
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
            __m256i acc_h, acc_l;
            __m256i m_bc = _mm256_set1_epi64x(Tbl::M[0][i].fe);
            Goldilocks::mult_avx_72(acc_h, acc_l, st[0], m_bc);
            for (uint32_t j = 1; j < SPONGE_WIDTH; ++j) {
                m_bc = _mm256_set1_epi64x(Tbl::M[j][i].fe);
                __m256i t_h, t_l;
                Goldilocks::mult_avx_72(t_h, t_l, st[j], m_bc);
                Goldilocks::add_avx(acc_l, acc_l, t_l);
                acc_h = _mm256_add_epi64(acc_h, t_h);
            }
            Goldilocks::reduce_avx_96_64(new_st[i], acc_h, acc_l);
        }
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) st[i] = new_st[i];
    };
    auto matmul_P = [&]() {
        __m256i new_st[SPONGE_WIDTH];
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
            __m256i p_bc = _mm256_set1_epi64x(Tbl::P[0][i].fe);
            Goldilocks::mult_avx(new_st[i], st[0], p_bc);
            for (uint32_t j = 1; j < SPONGE_WIDTH; ++j) {
                p_bc = _mm256_set1_epi64x(Tbl::P[j][i].fe);
                __m256i term;
                Goldilocks::mult_avx(term, st[j], p_bc);
                Goldilocks::add_avx(new_st[i], new_st[i], term);
            }
        }
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) st[i] = new_st[i];
    };
    auto pow7_all = [&]() {
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) element_pow7_avx(st[i]);
    };
    auto add_C = [&](uint32_t base) {
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
            __m256i c_bc = _mm256_set1_epi64x(Tbl::C[base + i].fe);
            Goldilocks::add_avx(st[i], st[i], c_bc);
        }
    };

    // First half of full rounds, minus the transition round.
    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS - 1; ++r) {
        pow7_all();
        add_C((r + 1) * SPONGE_WIDTH);
        matmul_M();
    }

    // Transition: pow7 + add(C) + matmul(P).
    pow7_all();
    add_C(HALF_N_FULL_ROUNDS * SPONGE_WIDTH);
    matmul_P();

    // Partial rounds.
    for (uint32_t r = 0; r < N_PARTIAL_ROUNDS; ++r) {
        // s0_bc = pow7(old st[0]) + C[partial_r]
        __m256i s0_bc = st[0];
        element_pow7_avx(s0_bc);
        __m256i c_bc = _mm256_set1_epi64x(
            Tbl::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r].fe);
        Goldilocks::add_avx(s0_bc, s0_bc, c_bc);
        st[0] = s0_bc;

        // new_st0 = Σ_j S_col[j] * st[j]   (dot with column)
        const Goldilocks::Element *S_col =
            &Tbl::S[(SPONGE_WIDTH * 2 - 1) * r];
        __m256i new_st0;
        __m256i col_bc = _mm256_set1_epi64x(S_col[0].fe);
        Goldilocks::mult_avx(new_st0, st[0], col_bc);
        for (uint32_t j = 1; j < SPONGE_WIDTH; ++j) {
            col_bc = _mm256_set1_epi64x(S_col[j].fe);
            __m256i term;
            Goldilocks::mult_avx(term, st[j], col_bc);
            Goldilocks::add_avx(new_st0, new_st0, term);
        }

        // Rank-1 update: st[i] += s0_bc * S_row_base[i]  for i=1..W-1.
        const Goldilocks::Element *S_row_base =
            &Tbl::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1];
        for (uint32_t i = 1; i < SPONGE_WIDTH; ++i) {
            __m256i row_bc = _mm256_set1_epi64x(S_row_base[i].fe);
            __m256i term;
            Goldilocks::mult_avx(term, s0_bc, row_bc);
            Goldilocks::add_avx(st[i], st[i], term);
        }

        st[0] = new_st0;
    }

    // Second half of full rounds, minus the final one.
    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS - 1; ++r) {
        pow7_all();
        add_C((HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH);
        matmul_M();
    }

    // Final: pow7 + matmul(M), no constant add.
    pow7_all();
    matmul_M();

    // Strided store.
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        Goldilocks::store_avx(&state[i], SPONGE_WIDTH, st[i]);
    }

}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::permuteTrunc_batch_avx(
    Goldilocks::Element (&state)[4 * CAPACITY],
    const Goldilocks::Element (&input)[4 * SPONGE_WIDTH])
{
    Goldilocks::Element aux[4 * SPONGE_WIDTH];
    permute_batch_avx(aux, input);
    std::memcpy(&state[0],           &aux[0],                 CAPACITY * sizeof(Goldilocks::Element));
    std::memcpy(&state[CAPACITY],    &aux[SPONGE_WIDTH],      CAPACITY * sizeof(Goldilocks::Element));
    std::memcpy(&state[2*CAPACITY],  &aux[2 * SPONGE_WIDTH],  CAPACITY * sizeof(Goldilocks::Element));
    std::memcpy(&state[3*CAPACITY],  &aux[3 * SPONGE_WIDTH],  CAPACITY * sizeof(Goldilocks::Element));
}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::linear_hash_batch_avx(
    Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[4 * SPONGE_WIDTH];

    while (remaining)
    {
        if (remaining == size)
        {
            for (uint64_t i = 0; i < 4; ++i) {
                memset(&state[i * SPONGE_WIDTH + RATE], 0, CAPACITY * sizeof(Goldilocks::Element));
            }
        }
        else
        {
            for (uint64_t i = 0; i < 4; ++i) {
                memmove(&state[i * SPONGE_WIDTH + RATE], &state[i * SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
            }
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        for (uint64_t i = 0; i < 4; ++i) {
            memset(&state[i * SPONGE_WIDTH + n], 0, (RATE - n) * sizeof(Goldilocks::Element));
            std::memcpy(&state[i * SPONGE_WIDTH], &input[i * size + (size - remaining)], n * sizeof(Goldilocks::Element));
        }
        permute_batch_avx(state, state);
        remaining -= n;
    }
    if (size > 0)
    {
        for (uint64_t i = 0; i < 4; ++i) {
            std::memcpy(&output[i * CAPACITY], &state[i * SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
        }
    }
    else
    {
        memset(output, 0, 4 * CAPACITY * sizeof(Goldilocks::Element));
    }
}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::merkletree_batch_avx(
    Goldilocks::Element *tree, Goldilocks::Element *input,
    uint64_t num_cols, uint64_t num_rows, uint64_t arity,
    int num_threads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    Goldilocks::Element *cursor = tree;
    if (num_threads == 0)
        num_threads = omp_get_max_threads();

#pragma omp parallel for num_threads(num_threads)
    for (uint64_t i = 0; i < num_rows; i += 4)
    {
        linear_hash_batch_avx(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
    }

    // Build the merkle tree (arity-generic).
    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
        {
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
        }

    #pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < nextN; i += 4)
        {
            // Read arity*CAPACITY child hashes per parent, zero-pad the tail.
            const uint64_t STRIDE = arity * CAPACITY;
            if (nextN - i < 4) {
                Goldilocks::Element pol_input[SPONGE_WIDTH];
                memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for (int j = 0; j < int(nextN - i); j++) {
                    std::memcpy(pol_input, &cursor[nextIndex + (i + j) * STRIDE], STRIDE * sizeof(Goldilocks::Element));
                    if constexpr (SPONGE_WIDTH_T == 12) {
                        permuteTrunc_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                    } else {
                        permuteTrunc_seq((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                    }
                }
            } else {
                Goldilocks::Element pol_input[4 * SPONGE_WIDTH];
                memset(pol_input, 0, 4 * SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for (uint32_t j = 0; j < 4; j++)
                {
                    std::memcpy(pol_input + j * SPONGE_WIDTH, &cursor[nextIndex + (i + j) * STRIDE], STRIDE * sizeof(Goldilocks::Element));
                }
                permuteTrunc_batch_avx((Goldilocks::Element(&)[4 * CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
            }
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

#endif // __AVX2__

// ---------------------------------------------------------------------------
// AVX512 (8-sponge batch only).
// ---------------------------------------------------------------------------

#ifdef __AVX512__

// 8-sponge AVX512 batch permute. 12 __m512i state registers, each holding one
// state element across 8 sponges (strided layout).
template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::permute_batch_avx512(
    Goldilocks::Element *state, const Goldilocks::Element *input)
{

    std::memcpy(state, input, 8 * SPONGE_WIDTH * sizeof(Goldilocks::Element));

    using Tbl = PoseidonGoldilocksConstants::Poseidon1Tables<SPONGE_WIDTH_T>;

    __m512i st[SPONGE_WIDTH];
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        Goldilocks::load_avx512(st[i], &state[i], SPONGE_WIDTH);
    }

    // Initial constant add.
    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        __m512i c_bc = _mm512_set1_epi64(Tbl::C[i].fe);
        Goldilocks::add_avx512(st[i], st[i], c_bc);
    }

    auto matmul_M = [&]() {
        // M12 entries fit in 8 bits → use 72-bit multiplier and accumulate
        // products pre-reduction (12× fewer reductions vs per-mult reduce).
        __m512i new_st[SPONGE_WIDTH];
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
            __m512i acc_h, acc_l;
            __m512i m_bc = _mm512_set1_epi64(Tbl::M[0][i].fe);
            Goldilocks::mult_avx512_72(acc_h, acc_l, st[0], m_bc);
            for (uint32_t j = 1; j < SPONGE_WIDTH; ++j) {
                m_bc = _mm512_set1_epi64(Tbl::M[j][i].fe);
                __m512i t_h, t_l;
                Goldilocks::mult_avx512_72(t_h, t_l, st[j], m_bc);
                Goldilocks::add_avx512(acc_l, acc_l, t_l);
                acc_h = _mm512_add_epi64(acc_h, t_h);
            }
            Goldilocks::reduce_avx512_96_64(new_st[i], acc_h, acc_l);
        }
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) st[i] = new_st[i];
    };
    auto matmul_P = [&]() {
        __m512i new_st[SPONGE_WIDTH];
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
            __m512i p_bc = _mm512_set1_epi64(Tbl::P[0][i].fe);
            Goldilocks::mult_avx512(new_st[i], st[0], p_bc);
            for (uint32_t j = 1; j < SPONGE_WIDTH; ++j) {
                p_bc = _mm512_set1_epi64(Tbl::P[j][i].fe);
                __m512i term;
                Goldilocks::mult_avx512(term, st[j], p_bc);
                Goldilocks::add_avx512(new_st[i], new_st[i], term);
            }
        }
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) st[i] = new_st[i];
    };
    auto pow7_all = [&]() {
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) element_pow7_avx512(st[i]);
    };
    auto add_C = [&](uint32_t base) {
        for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
            __m512i c_bc = _mm512_set1_epi64(Tbl::C[base + i].fe);
            Goldilocks::add_avx512(st[i], st[i], c_bc);
        }
    };

    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS - 1; ++r) {
        pow7_all();
        add_C((r + 1) * SPONGE_WIDTH);
        matmul_M();
    }

    pow7_all();
    add_C(HALF_N_FULL_ROUNDS * SPONGE_WIDTH);
    matmul_P();

    for (uint32_t r = 0; r < N_PARTIAL_ROUNDS; ++r) {
        __m512i s0_bc = st[0];
        element_pow7_avx512(s0_bc);
        __m512i c_bc = _mm512_set1_epi64(
            Tbl::C[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r].fe);
        Goldilocks::add_avx512(s0_bc, s0_bc, c_bc);
        st[0] = s0_bc;

        const Goldilocks::Element *S_col =
            &Tbl::S[(SPONGE_WIDTH * 2 - 1) * r];
        __m512i new_st0;
        __m512i col_bc = _mm512_set1_epi64(S_col[0].fe);
        Goldilocks::mult_avx512(new_st0, st[0], col_bc);
        for (uint32_t j = 1; j < SPONGE_WIDTH; ++j) {
            col_bc = _mm512_set1_epi64(S_col[j].fe);
            __m512i term;
            Goldilocks::mult_avx512(term, st[j], col_bc);
            Goldilocks::add_avx512(new_st0, new_st0, term);
        }

        const Goldilocks::Element *S_row_base =
            &Tbl::S[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1];
        for (uint32_t i = 1; i < SPONGE_WIDTH; ++i) {
            __m512i row_bc = _mm512_set1_epi64(S_row_base[i].fe);
            __m512i term;
            Goldilocks::mult_avx512(term, s0_bc, row_bc);
            Goldilocks::add_avx512(st[i], st[i], term);
        }

        st[0] = new_st0;
    }

    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS - 1; ++r) {
        pow7_all();
        add_C((HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH);
        matmul_M();
    }

    pow7_all();
    matmul_M();

    for (uint32_t i = 0; i < SPONGE_WIDTH; ++i) {
        Goldilocks::store_avx512(&state[i], SPONGE_WIDTH, st[i]);
    }

}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::permuteTrunc_batch_avx512(
    Goldilocks::Element (&state)[8 * CAPACITY],
    const Goldilocks::Element (&input)[8 * SPONGE_WIDTH])
{
    Goldilocks::Element aux[8 * SPONGE_WIDTH];
    permute_batch_avx512(aux, input);
    for (uint32_t lane = 0; lane < 8; ++lane)
    {
        std::memcpy(&state[lane * CAPACITY], &aux[lane * SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
    }
}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::linear_hash_batch_avx512(
    Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[8 * SPONGE_WIDTH];

    while (remaining)
    {
        if (remaining == size)
        {
            for (uint64_t i = 0; i < 8; ++i) {
                memset(&state[i * SPONGE_WIDTH + RATE], 0, CAPACITY * sizeof(Goldilocks::Element));
            }
        }
        else
        {
            for (uint64_t i = 0; i < 8; ++i) {
                memmove(&state[i * SPONGE_WIDTH + RATE], &state[i * SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
            }
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        for (uint64_t i = 0; i < 8; ++i) {
            memset(&state[i * SPONGE_WIDTH + n], 0, (RATE - n) * sizeof(Goldilocks::Element));
            std::memcpy(&state[i * SPONGE_WIDTH], &input[i * size + (size - remaining)], n * sizeof(Goldilocks::Element));
        }
        permute_batch_avx512(state, state);
        remaining -= n;
    }
    if (size > 0)
    {
        for (uint64_t i = 0; i < 8; ++i) {
            std::memcpy(&output[i * CAPACITY], &state[i * SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
        }
    }
    else
    {
        memset(output, 0, 8 * CAPACITY * sizeof(Goldilocks::Element));
    }
}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::merkletree_batch_avx512(
    Goldilocks::Element *tree, Goldilocks::Element *input,
    uint64_t num_cols, uint64_t num_rows, uint64_t arity,
    int num_threads, uint64_t dim)
{
    if (num_rows == 0)
    {
        return;
    }
    Goldilocks::Element *cursor = tree;
    if (num_threads == 0)
        num_threads = omp_get_max_threads();

#pragma omp parallel for num_threads(num_threads)
    for (uint64_t i = 0; i < num_rows; i += 8)
    {
        linear_hash_batch_avx512(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
    }

    // Build the merkle tree (arity-generic).
    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
        {
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
        }

    #pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < nextN; i += 8)
        {
            const uint64_t STRIDE = arity * CAPACITY;
            if (nextN - i < 8) {
                Goldilocks::Element pol_input[SPONGE_WIDTH];
                memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for (uint32_t j = 0; j < uint32_t(nextN - i); j++) {
                    std::memcpy(pol_input, &cursor[nextIndex + (i + j) * STRIDE], STRIDE * sizeof(Goldilocks::Element));
                    if constexpr (SPONGE_WIDTH_T == 12) {
                        permuteTrunc_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                    } else {
                        permuteTrunc_seq((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                    }
                }
            } else {
                Goldilocks::Element pol_input[8 * SPONGE_WIDTH];
                memset(pol_input, 0, 8 * SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for (uint32_t j = 0; j < 8; j++)
                {
                    std::memcpy(pol_input + j * SPONGE_WIDTH, &cursor[nextIndex + (i + j) * STRIDE], STRIDE * sizeof(Goldilocks::Element));
                }
                permuteTrunc_batch_avx512((Goldilocks::Element(&)[8 * CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
            }
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

#endif // __AVX512__

// ---------------------------------------------------------------------------
// Explicit template instantiation.
// ---------------------------------------------------------------------------

template class PoseidonGoldilocks<8>;
template class PoseidonGoldilocks<12>;
template class PoseidonGoldilocks<16>;
