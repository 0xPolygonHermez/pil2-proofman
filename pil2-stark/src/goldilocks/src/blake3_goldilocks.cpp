#include "blake3_goldilocks.hpp"
#include "blake3_core.hpp"

#include <cstring>
#include <cmath>
#include <atomic>
#include <stdexcept>
#include <omp.h>

void Blake3Goldilocks::linearHash(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t dig[4];
    blake3core::hash_le64(reinterpret_cast<const uint64_t *>(input), (uint32_t)size, dig);
    for (int i = 0; i < 4; ++i) output[i].fe = dig[i];
}

void Blake3Goldilocks::permuteTrunc(Goldilocks::Element (&output)[CAPACITY], const Goldilocks::Element (&input)[8])
{
    uint64_t dig[4];
    blake3core::hash_le64(reinterpret_cast<const uint64_t *>(&input[0]), 8, dig);
    for (int i = 0; i < 4; ++i) output[i].fe = dig[i];
}

void Blake3Goldilocks::permute(Goldilocks::Element (&output)[8], const Goldilocks::Element (&input)[8])
{
    uint64_t out8[8];
    blake3core::permute8(reinterpret_cast<const uint64_t *>(&input[0]), out8);
    for (int i = 0; i < 8; ++i) output[i].fe = out8[i];
}

void Blake3Goldilocks::permuteTranscript(Goldilocks::Element *output, const Goldilocks::Element *input, uint64_t width)
{
    blake3core::permute_xof(reinterpret_cast<const uint64_t *>(input), (uint32_t)width,
                            reinterpret_cast<uint64_t *>(output));
}

void Blake3Goldilocks::merkletree(Goldilocks::Element *tree, Goldilocks::Element *input,
                                  uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                                  int num_threads, uint64_t dim)
{
    if (num_rows == 0) return;
    Goldilocks::Element *cursor = tree;
    if (num_threads == 0) num_threads = omp_get_max_threads();

#pragma omp parallel for num_threads(num_threads)
    for (uint64_t i = 0; i < num_rows; i++)
        linearHash(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);

    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));

#pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < nextN; i++)
        {
            uint64_t dig[4];
            blake3core::hash_le64(reinterpret_cast<const uint64_t *>(&cursor[nextIndex + i * (arity * CAPACITY)]),
                                  (uint32_t)(arity * CAPACITY), dig);
            Goldilocks::Element *outp = &cursor[nextIndex + (pending + extraZeros + i) * CAPACITY];
            for (int j = 0; j < 4; ++j) outp[j].fe = dig[j];
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

void Blake3Goldilocks::merkletreeReduce(Goldilocks::Element *root, Goldilocks::Element *input,
                                        uint64_t num_elements, uint64_t arity)
{
    uint64_t numNodes = num_elements;
    uint64_t nodesLevel = num_elements;
    while (nodesLevel > 1)
    {
        uint64_t extraZeros = (arity - (nodesLevel % arity)) % arity;
        numNodes += extraZeros;
        uint64_t nextN = (nodesLevel + (arity - 1)) / arity;
        numNodes += nextN;
        nodesLevel = nextN;
    }

    Goldilocks::Element *cursor = new Goldilocks::Element[numNodes * CAPACITY];
    std::memcpy(cursor, input, num_elements * CAPACITY * sizeof(Goldilocks::Element));

    uint64_t pending = num_elements;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));

        for (uint64_t i = 0; i < nextN; i++)
        {
            uint64_t dig[4];
            blake3core::hash_le64(reinterpret_cast<const uint64_t *>(&cursor[nextIndex + i * (arity * CAPACITY)]),
                                  (uint32_t)(arity * CAPACITY), dig);
            Goldilocks::Element *outp = &cursor[nextIndex + (pending + extraZeros + i) * CAPACITY];
            for (int j = 0; j < 4; ++j) outp[j].fe = dig[j];
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }

    std::memcpy(root, &cursor[nextIndex], CAPACITY * sizeof(Goldilocks::Element));
    delete[] cursor;
}

void Blake3Goldilocks::grinding(uint64_t &nonce, const uint64_t *in, const uint32_t n_bits)
{
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

        // STARK grinding contract: state[0..2] = challenge, state[3] = nonce, rest 0.
        uint64_t state[8] = {in[0], in[1], in[2], 0, 0, 0, 0, 0};
        uint64_t out[8];

        for (uint64_t i = tid; i < N; i += nthr)
        {
            if ((poll++ & POLL_MASK) == 0)
                local_found = found.load(std::memory_order_relaxed);
            if (i >= local_found) break;

            state[3] = i;
            blake3core::permute8(state, out);
            if (out[0] < level)
            {
#pragma omp critical(grinding_update_b3)
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
        throw std::runtime_error("Blake3Goldilocks::grinding: could not find a valid nonce");
}
