#include "sha256_goldilocks.hpp"
#include "sha256_core.hpp"

#include <cstring>
#include <cmath>
#include <atomic>
#include <stdexcept>
#include <omp.h>

void Sha256Goldilocks::linearHash(Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    uint64_t dig[4];
    sha256core::hash_le64(reinterpret_cast<const uint64_t *>(input), (uint32_t)size, dig);
    for (int i = 0; i < 4; ++i) output[i].fe = dig[i];
}

void Sha256Goldilocks::nodeHash(Goldilocks::Element *output, const Goldilocks::Element *input, uint64_t size)
{
    if (size == 0 || size % sha256core::BLOCK_U64 != 0)
        throw std::runtime_error("Sha256Goldilocks::nodeHash: node width must be a positive multiple of 8 "
                                 "(arity * CAPACITY); got " + std::to_string(size));
    uint64_t dig[4];
    sha256core::node_hash(reinterpret_cast<const uint64_t *>(input), (uint32_t)size, dig);
    for (int i = 0; i < 4; ++i) output[i].fe = dig[i];
}

void Sha256Goldilocks::permuteTrunc(Goldilocks::Element (&output)[CAPACITY], const Goldilocks::Element (&input)[8])
{
    uint64_t dig[4];
    sha256core::hash_le64(reinterpret_cast<const uint64_t *>(&input[0]), 8, dig);
    for (int i = 0; i < 4; ++i) output[i].fe = dig[i];
}

void Sha256Goldilocks::permute(Goldilocks::Element (&output)[8], const Goldilocks::Element (&input)[8])
{
    uint64_t dig[4];
    sha256core::grind_hash(reinterpret_cast<const uint64_t *>(&input[0]), dig);
    for (int i = 0; i < 4; ++i) output[i].fe = dig[i];
    // No fifth..eighth digest word; cleared so a stray read is at least stable.
    for (int i = 4; i < 8; ++i) output[i] = Goldilocks::zero();
}

void Sha256Goldilocks::merkletree(Goldilocks::Element *tree, Goldilocks::Element *input,
                                  uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                                  int num_threads, uint64_t dim)
{
    if (num_rows == 0) return;
    Goldilocks::Element *cursor = tree;
    if (num_threads == 0) num_threads = omp_get_max_threads();

    // Leaves: variable width, so FIPS.
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

        // Internal nodes: fixed width, so the domain-separated hash.
#pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < nextN; i++)
        {
            uint64_t dig[4];
            sha256core::node_hash(reinterpret_cast<const uint64_t *>(&cursor[nextIndex + i * (arity * CAPACITY)]),
                                  (uint32_t)(arity * CAPACITY), dig);
            Goldilocks::Element *outp = &cursor[nextIndex + (pending + extraZeros + i) * CAPACITY];
            for (int j = 0; j < 4; ++j) outp[j].fe = dig[j];
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
}

void Sha256Goldilocks::merkletreeReduce(Goldilocks::Element *root, Goldilocks::Element *input,
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
            sha256core::node_hash(reinterpret_cast<const uint64_t *>(&cursor[nextIndex + i * (arity * CAPACITY)]),
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

void Sha256Goldilocks::grinding(uint64_t &nonce, const uint64_t *in, const uint32_t n_bits)
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
        uint64_t out[4];

        for (uint64_t i = tid; i < N; i += nthr)
        {
            if ((poll++ & POLL_MASK) == 0)
                local_found = found.load(std::memory_order_relaxed);
            if (i >= local_found) break;

            state[3] = i;
            sha256core::grind_hash(state, out);
            if (out[0] < level)
            {
#pragma omp critical(grinding_update_sha256)
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
        throw std::runtime_error("Sha256Goldilocks::grinding: could not find a valid nonce");
}
