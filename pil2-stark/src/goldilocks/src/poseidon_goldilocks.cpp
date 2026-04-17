#include "poseidon_goldilocks.hpp"
#include <math.h> /* floor */
#include <omp.h>
#include <cassert>

// ---------------------------------------------------------------------------
// Poseidon v1 over Goldilocks — port of pil2-stark 0.14.0.
//
// Math is byte-identical to pil2-stark 0.14.0 poseidon_goldilocks.cpp.
// Structural changes from 0.14.0:
//   * Class is templated over SPONGE_WIDTH_T (only W=12 instantiated).
//   * Constants live in namespace PoseidonGoldilocksConstants with width-
//     suffixed names (C12/M12/M_12/P12/P_12/S12).
//   * merkletree now takes an `arity` parameter (mirrors Poseidon2's shape).
//   * Scalar batch paths and single-sponge AVX512 are intentionally omitted
//     (Poseidon2 doesn't expose them either — keeping the public API aligned).
//   * Batch AVX / batch AVX512 variants in Poseidon v1 didn't exist in 0.14.0.
//     They're provided here as per-lane loops over the single-sponge routine,
//     which keeps results byte-exact with the committed goldens while
//     matching the Poseidon2 batch API contract.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Scalar
// ---------------------------------------------------------------------------

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::permute_seq(
    Goldilocks::Element (&state)[SPONGE_WIDTH],
    const Goldilocks::Element (&input)[SPONGE_WIDTH])
{
    const int length = SPONGE_WIDTH * sizeof(Goldilocks::Element);
    std::memcpy(state, input, length);

    add_(state, &(PoseidonGoldilocksConstants::C12[0]));
    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7add_(state, &(PoseidonGoldilocksConstants::C12[(r + 1) * SPONGE_WIDTH]));
        mvp_(state, PoseidonGoldilocksConstants::M12);
    }
    pow7add_(state, &(PoseidonGoldilocksConstants::C12[(HALF_N_FULL_ROUNDS * SPONGE_WIDTH)]));
    mvp_(state, PoseidonGoldilocksConstants::P12);

    for (uint32_t r = 0; r < N_PARTIAL_ROUNDS; r++)
    {
        pow7(state[0]);
        state[0] = state[0] + PoseidonGoldilocksConstants::C12[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + r];
        Goldilocks::Element s0 = dot_(state, &(PoseidonGoldilocksConstants::S12[(SPONGE_WIDTH * 2 - 1) * r]));
        Goldilocks::Element W_[SPONGE_WIDTH];
        prod_(W_, state[0], &(PoseidonGoldilocksConstants::S12[(SPONGE_WIDTH * 2 - 1) * r + SPONGE_WIDTH - 1]));
        add_(state, W_);
        state[0] = s0;
    }

    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS - 1; r++)
    {
        pow7add_(state, &(PoseidonGoldilocksConstants::C12[(HALF_N_FULL_ROUNDS + 1) * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]));
        mvp_(state, PoseidonGoldilocksConstants::M12);
    }
    pow7_(&(state[0]));
    mvp_(state, PoseidonGoldilocksConstants::M12);
}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::linear_hash_seq(
    Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t remaining = size;
    Goldilocks::Element state[SPONGE_WIDTH];

    // pil2-stark 0.14.0 treats small inputs (size <= CAPACITY) as pass-through,
    // not hashed. Preserved verbatim for interop with committed goldens.
    if (size <= CAPACITY)
    {
        std::memcpy(output, input, size * sizeof(Goldilocks::Element));
        std::memset(&output[size], 0, (CAPACITY - size) * sizeof(Goldilocks::Element));
        return; // no need to hash
    }
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

    // Build the merkle tree (arity-generic reduction — mirrors Poseidon2).
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

            // Arity-correct stride: read arity*CAPACITY child hashes (each of
            // CAPACITY elements) and zero-pad the remaining SPONGE_WIDTH slots.
            // This differs from Poseidon2's pattern (which assumes arity*CAP ==
            // SPONGE_WIDTH), so that arity=2 at W=12 reproduces 0.14.0's
            // binary-merkle root bit-for-bit.
            std::memcpy(pol_input, &cursor[nextIndex + i * (arity * CAPACITY)], (arity * CAPACITY) * sizeof(Goldilocks::Element));

            compress_seq((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
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
    // Only arity=2 is validated; see comment in merkletree() in the header.
    assert(arity == 2 && "PoseidonGoldilocks::merkletreeReduce: only arity == 2 is supported");
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

            // Arity-correct stride: read arity*CAPACITY child hashes (each of
            // CAPACITY elements) and zero-pad the remaining SPONGE_WIDTH slots.
            // This differs from Poseidon2's pattern (which assumes arity*CAP ==
            // SPONGE_WIDTH), so that arity=2 at W=12 reproduces 0.14.0's
            // binary-merkle root bit-for-bit.
            std::memcpy(pol_input, &cursor[nextIndex + i * (arity * CAPACITY)], (arity * CAPACITY) * sizeof(Goldilocks::Element));

            compress_seq((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
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
    uint64_t checkChunk = omp_get_max_threads() * 512;
    uint64_t level   = uint64_t(1) << (64 - n_bits);
    uint64_t *chunkIdxs = new uint64_t[omp_get_max_threads()];
    uint64_t offset = 0;
    nonce = UINT64_MAX;

    for (int i = 0; i < omp_get_max_threads(); ++i)
    {
        chunkIdxs[i] = UINT64_MAX;
    }

    // we are trying (1 << n_bits) * 512 * num_threads possibilities maximum
    for (int k = 0; k < (1 << n_bits); ++k)
    {

        #pragma omp parallel for
        for (uint64_t i = 0; i < checkChunk; i++) {
            if (chunkIdxs[omp_get_thread_num()] != UINT64_MAX)
                continue;

            Goldilocks::Element state[SPONGE_WIDTH];
            std::memcpy(state, in, (SPONGE_WIDTH - 1) * sizeof(Goldilocks::Element));
            state[SPONGE_WIDTH - 1] = Goldilocks::fromU64(offset + i);
            permute_seq(state, state);
            if (state[0].fe < level) {
                chunkIdxs[omp_get_thread_num()] = offset + i;
            }
        }

        for (int i = 0; i < omp_get_max_threads(); ++i)
        {
            if (chunkIdxs[i] != UINT64_MAX)
            {
                nonce = chunkIdxs[i];
                break;
            }
        }

        if (nonce != UINT64_MAX)
            break;

        offset += checkChunk;
    }
    if (nonce == UINT64_MAX)
    {
        throw std::runtime_error("PoseidonGoldilocks::grinding: could not find a valid nonce");
    }
    delete[] chunkIdxs;
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

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::compress_avx(
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

    // Matches linear_hash_seq's pass-through branch (byte-identical goldens).
    if (size <= CAPACITY)
    {
        std::memcpy(output, input, size * sizeof(Goldilocks::Element));
        std::memset(&output[size], 0, (CAPACITY - size) * sizeof(Goldilocks::Element));
        return; // no need to hash
    }
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

            // Arity-correct stride: read arity*CAPACITY child hashes (each of
            // CAPACITY elements) and zero-pad the remaining SPONGE_WIDTH slots.
            // This differs from Poseidon2's pattern (which assumes arity*CAP ==
            // SPONGE_WIDTH), so that arity=2 at W=12 reproduces 0.14.0's
            // binary-merkle root bit-for-bit.
            std::memcpy(pol_input, &cursor[nextIndex + i * (arity * CAPACITY)], (arity * CAPACITY) * sizeof(Goldilocks::Element));

            compress_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

// --- AVX2 4-lane batch -----------------------------------------------------
//
// Poseidon v1 in 0.14.0 didn't ship a batched AVX2 permute. To match the
// Poseidon2 batch API contract without introducing new algebra, each batch
// permutation is implemented as 4 independent single-sponge permute_avx calls.
// The result is bit-identical to the (already vectorized) single-sponge path
// and drops straight into the same merkletree_batch loop shape Poseidon2 uses.

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::permute_batch_avx(
    Goldilocks::Element *state, const Goldilocks::Element *input)
{
    for (uint32_t lane = 0; lane < 4; ++lane)
    {
        Goldilocks::Element (&s)[SPONGE_WIDTH] = (Goldilocks::Element(&)[SPONGE_WIDTH])state[lane * SPONGE_WIDTH];
        const Goldilocks::Element (&in)[SPONGE_WIDTH] = (const Goldilocks::Element(&)[SPONGE_WIDTH])input[lane * SPONGE_WIDTH];
        permute_avx(s, in);
    }
}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::compress_batch_avx(
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
            // Arity-correct stride matches the scalar and single-AVX paths:
            // read arity*CAPACITY child hashes per parent, zero-pad the tail.
            const uint64_t STRIDE = arity * CAPACITY;
            if (nextN - i < 4) {
                Goldilocks::Element pol_input[SPONGE_WIDTH];
                memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for (int j = 0; j < int(nextN - i); j++) {
                    std::memcpy(pol_input, &cursor[nextIndex + (i + j) * STRIDE], STRIDE * sizeof(Goldilocks::Element));
                    compress_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                }
            } else {
                Goldilocks::Element pol_input[4 * SPONGE_WIDTH];
                memset(pol_input, 0, 4 * SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for (uint32_t j = 0; j < 4; j++)
                {
                    std::memcpy(pol_input + j * SPONGE_WIDTH, &cursor[nextIndex + (i + j) * STRIDE], STRIDE * sizeof(Goldilocks::Element));
                }
                compress_batch_avx((Goldilocks::Element(&)[4 * CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
            }
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

#endif // __AVX2__

// ---------------------------------------------------------------------------
// AVX512 (8-lane batch only — single-sponge AVX512 intentionally unimplemented
// to match Poseidon2's public surface).
// ---------------------------------------------------------------------------

#ifdef __AVX512__

// Poseidon v1 didn't ship a batched AVX512 permute in 0.14.0. As for AVX2
// above, per-lane dispatch to the single-sponge (AVX2) permute keeps the
// results byte-exact with 0.14.0 while honoring the 8-lane batch contract.

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::permute_batch_avx512(
    Goldilocks::Element *state, const Goldilocks::Element *input)
{
    for (uint32_t lane = 0; lane < 8; ++lane)
    {
        Goldilocks::Element (&s)[SPONGE_WIDTH] = (Goldilocks::Element(&)[SPONGE_WIDTH])state[lane * SPONGE_WIDTH];
        const Goldilocks::Element (&in)[SPONGE_WIDTH] = (const Goldilocks::Element(&)[SPONGE_WIDTH])input[lane * SPONGE_WIDTH];
        permute_avx(s, in);
    }
}

template<uint32_t SPONGE_WIDTH_T>
void PoseidonGoldilocks<SPONGE_WIDTH_T>::compress_batch_avx512(
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
                    compress_avx((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + (i + j)) * CAPACITY], pol_input);
                }
            } else {
                Goldilocks::Element pol_input[8 * SPONGE_WIDTH];
                memset(pol_input, 0, 8 * SPONGE_WIDTH * sizeof(Goldilocks::Element));
                for (uint32_t j = 0; j < 8; j++)
                {
                    std::memcpy(pol_input + j * SPONGE_WIDTH, &cursor[nextIndex + (i + j) * STRIDE], STRIDE * sizeof(Goldilocks::Element));
                }
                compress_batch_avx512((Goldilocks::Element(&)[8 * CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY], pol_input);
            }
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

#endif // __AVX512__

// ---------------------------------------------------------------------------
// Explicit template instantiations.
// Only W=12 is shipped this iteration (matches pil2-stark 0.14.0 / Plonky2).
// ---------------------------------------------------------------------------
template class PoseidonGoldilocks<12>;
