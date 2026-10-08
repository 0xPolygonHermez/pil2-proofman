// Merkle trees stored without their leaf level (GPU commit trees): the row-range linear hash and
// level reduction they are built from, the builder itself, and the query-time rehash of the
// level-0 siblings. Every case is checked against the full tree built by H::merkletree.
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "../src/poseidon2_goldilocks.cuh"
#include "../src/poseidon_goldilocks.cuh"
#include "../src/blake3_goldilocks.cuh"
#include "../src/goldilocks_tooling.hpp"
#include "../src/merkle_noleaves.cuh"

namespace {

template <class H> void initHash() {
    uint32_t gpu = 0;
    cudaGetDevice((int *)&gpu);
    H::initConstants(&gpu, 1);
}
template <> void initHash<Blake3GoldilocksGPU>() {}

std::vector<uint64_t> randomTrace(uint64_t nCols, uint64_t nRows, uint64_t seed) {
    std::vector<uint64_t> h(std::max<uint64_t>(nCols, 1) * nRows);
    for (uint64_t i = 0; i < h.size(); i++) h[i] = ((i + seed) * 0x9E3779B97F4A7C15ull) % 0xFFFFFFFF00000001ull;
    return h;
}

uint64_t *toDevice(const std::vector<uint64_t> &h) {
    uint64_t *d;
    cudaMalloc(&d, h.size() * 8);
    cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice);
    return d;
}

std::vector<uint64_t> toHost(const uint64_t *d, uint64_t n) {
    std::vector<uint64_t> h(n);
    cudaMemcpy(h.data(), d, n * 8, cudaMemcpyDeviceToHost);
    return h;
}

// Rows [r0, r0 + cnt) of a ColMajor nRows-tall matrix, hashed through the leading dimension,
// must equal those rows' digests from a whole-matrix hash.
template <class H>
void checkLinearHashRange(uint64_t nCols, uint64_t nRows, uint64_t r0, uint64_t cnt) {
    initHash<H>();
    uint64_t *d_in = toDevice(randomTrace(nCols, nRows, 1));
    uint64_t *d_full, *d_part;
    cudaMalloc(&d_full, nRows * 32);
    cudaMalloc(&d_part, cnt * 32);
    H::linearHash(d_full, d_in, nCols, nRows, Layout::ColMajor, 0);
    H::linearHash(d_part, d_in + r0, nCols, cnt, Layout::ColMajor, 0, nRows);
    auto full = toHost(d_full, nRows * 4), part = toHost(d_part, cnt * 4);
    for (uint64_t i = 0; i < cnt * 4; i++) ASSERT_EQ(part[i], full[r0 * 4 + i]) << "digest word " << i;
    cudaFree(d_in); cudaFree(d_full); cudaFree(d_part);
}

// linearHash + reduceLevels over the leaves must rebuild the whole tree byte for byte.
template <class H>
void checkReduceLevels(uint32_t arity, uint64_t nCols, uint64_t nRows) {
    initHash<H>();
    uint64_t *d_in = toDevice(randomTrace(nCols, nRows, 2));
    const uint64_t n = getTreeNumElements(nRows, arity);
    uint64_t *d_ref, *d_tree;
    cudaMalloc(&d_ref, n * 8);
    cudaMalloc(&d_tree, n * 8);
    H::merkletree(arity, d_ref, d_in, nCols, nRows, Layout::ColMajor, 0);
    H::linearHash(d_tree, d_in, nCols, nRows, Layout::ColMajor, 0);
    H::reduceLevels(arity, d_tree, nRows, 0);
    auto ref = toHost(d_ref, n), tree = toHost(d_tree, n);
    for (uint64_t i = 0; i < n; i++) ASSERT_EQ(tree[i], ref[i]) << "tree word " << i;
    cudaFree(d_in); cudaFree(d_ref); cudaFree(d_tree);
}

// The stored tree (level 1 upward) must equal the full tree minus its padded leaf level.
template <class H>
void checkNoLeaves(uint32_t arity, uint64_t nCols, uint64_t nRows, uint64_t maxChunkRows) {
    initHash<H>();
    uint64_t *d_in = toDevice(randomTrace(nCols, nRows, 3));
    const uint64_t fullN = getTreeNumElements(nRows, arity), stN = getTreeNumElementsNoLeaves(nRows, arity);
    const uint64_t leafPad = nRows + (arity - nRows % arity) % arity;
    ASSERT_EQ(stN, fullN - leafPad * 4);
    uint64_t *d_full, *d_st;
    cudaMalloc(&d_full, fullN * 8);
    cudaMalloc(&d_st, stN * 8);
    cudaMemset(d_st, 0xAB, stN * 8);  // no stale zeros to hide a missing write
    H::merkletree(arity, d_full, d_in, nCols, nRows, Layout::ColMajor, 0);
    merkletreeNoLeaves<H>(arity, d_st, d_in, nCols, nRows, Layout::ColMajor, 0, maxChunkRows);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto full = toHost(d_full, fullN), st = toHost(d_st, stN);
    for (uint64_t i = 0; i < stN; i++) ASSERT_EQ(st[i], full[leafPad * 4 + i]) << "stored word " << i;
    cudaFree(d_in); cudaFree(d_full); cudaFree(d_st);
}

// Level-0 siblings rebuilt from the rows must equal the full tree's leaf digests (zeros past the
// last row, where the full tree pads the leaf level).
template <class H>
void checkLeafSiblings(uint32_t a, uint64_t nCols, uint64_t nRows, std::vector<uint64_t> qs) {
    initHash<H>();
    uint64_t *d_in = toDevice(randomTrace(nCols, nRows, 4));
    const uint64_t fullN = getTreeNumElements(nRows, a);
    uint64_t *d_full;
    cudaMalloc(&d_full, fullN * 8);
    H::merkletree(a, d_full, d_in, nCols, nRows, Layout::ColMajor, 0);
    const uint64_t nq = qs.size(), width = nCols, bufW = std::max<uint64_t>(nCols, 1) + 64;
    uint64_t *d_q = toDevice(qs);
    uint64_t *d_scr, *d_buf;
    cudaMalloc(&d_scr, leafSiblingsScratchWords(nq, a, nCols) * 8);
    cudaMalloc(&d_buf, nq * bufW * 8);
    cudaMemset(d_buf, 0xAB, nq * bufW * 8);
    leafSiblingsNoLeaves<H>((gl64_t *)d_in, nCols, nRows, Layout::ColMajor, d_q, nq, a, (gl64_t *)d_scr,
                            (gl64_t *)d_buf, bufW, width, 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto buf = toHost(d_buf, nq * bufW), full = toHost(d_full, fullN);
    for (uint64_t q = 0; q < nq; q++) {
        const uint64_t idx = qs[q], si = idx - idx % a;
        uint64_t j = 0;
        for (uint64_t i = 0; i < a; i++) {
            if (si + i == idx) continue;
            for (int w = 0; w < 4; w++) {
                const uint64_t exp = (si + i < nRows) ? full[(si + i) * 4 + w] : 0;
                ASSERT_EQ(buf[q * bufW + width + j * 4 + w], exp) << "q=" << q << " sibling row " << si + i;
            }
            j++;
        }
    }
    cudaFree(d_in); cudaFree(d_full); cudaFree(d_q); cudaFree(d_scr); cudaFree(d_buf);
}

} // namespace

using P2A2 = Poseidon2GoldilocksGPU<8>;
using P2A3 = Poseidon2GoldilocksGPU<12>;
using P2A4 = Poseidon2GoldilocksGPU<16>;
using P1A2 = PoseidonGoldilocksGPU<8>;
using P1A4 = PoseidonGoldilocksGPU<16>;
using B3 = Blake3GoldilocksGPU;

TEST(merkle_noleaves, linear_hash_range_pos2) { checkLinearHashRange<P2A2>(37, 1024, 256, 300); }
TEST(merkle_noleaves, linear_hash_range_pos1) { checkLinearHashRange<P1A2>(37, 1024, 256, 300); }
TEST(merkle_noleaves, linear_hash_range_b3) { checkLinearHashRange<B3>(37, 1024, 256, 300); }

TEST(merkle_noleaves, reduce_levels_pos2_a2) { checkReduceLevels<P2A2>(2, 37, 4096); }
TEST(merkle_noleaves, reduce_levels_pos2_a3) { checkReduceLevels<P2A3>(3, 5, 4096); }
TEST(merkle_noleaves, reduce_levels_pos1_a4) { checkReduceLevels<P1A4>(4, 37, 4096); }
TEST(merkle_noleaves, reduce_levels_b3_a2) { checkReduceLevels<B3>(2, 37, 4096); }

TEST(merkle_noleaves, tree_pos2_a2) { checkNoLeaves<P2A2>(2, 37, 4096, 0); }
TEST(merkle_noleaves, tree_pos2_a2_chunks) { checkNoLeaves<P2A2>(2, 37, 4096, 250); }  // uneven last chunk
TEST(merkle_noleaves, tree_pos2_a4) { checkNoLeaves<P2A4>(4, 5, 4096, 300); }
TEST(merkle_noleaves, tree_pos2_a3_pad) { checkNoLeaves<P2A3>(3, 5, 4096, 301); }      // 4096 % 3 != 0
TEST(merkle_noleaves, tree_pos1_a2) { checkNoLeaves<P1A2>(2, 37, 4096, 250); }
TEST(merkle_noleaves, tree_pos1_a4) { checkNoLeaves<P1A4>(4, 37, 4096, 300); }
TEST(merkle_noleaves, tree_b3_a2) { checkNoLeaves<B3>(2, 37, 4096, 250); }
TEST(merkle_noleaves, tree_pos2_zero_cols) { checkNoLeaves<P2A2>(2, 0, 1024, 0); }
TEST(merkle_noleaves, tree_pos2_a2_default_chunk_large) { checkNoLeaves<P2A2>(2, 3, 1 << 16, 0); }

TEST(merkle_noleaves, siblings_pos2_a2) { checkLeafSiblings<P2A2>(2, 37, 4096, {0, 1, 2047, 4095}); }
TEST(merkle_noleaves, siblings_pos2_a3_pad) { checkLeafSiblings<P2A3>(3, 5, 4096, {0, 4095, 4094, 7}); }
TEST(merkle_noleaves, siblings_pos2_a4) { checkLeafSiblings<P2A4>(4, 37, 4096, {3, 1000, 4095}); }
TEST(merkle_noleaves, siblings_pos1_a4) { checkLeafSiblings<P1A4>(4, 37, 4096, {3, 1000}); }
TEST(merkle_noleaves, siblings_b3_a2) { checkLeafSiblings<B3>(2, 37, 4096, {0, 4095, 17}); }
TEST(merkle_noleaves, siblings_zero_cols) { checkLeafSiblings<P2A2>(2, 0, 1024, {5}); }
TEST(merkle_noleaves, siblings_many_queries) {
    std::vector<uint64_t> qs;
    for (uint64_t i = 0; i < 211; i++) qs.push_back((i * 2654435761ull) % 8192);
    checkLeafSiblings<P2A2>(2, 152, 8192, qs);
}
// The smallest trees StarkInfo drops the leaf level for (NExtended = arity^3).
TEST(merkle_noleaves, tree_min_a2) { checkNoLeaves<P2A2>(2, 5, 8, 0); }
TEST(merkle_noleaves, tree_min_a3) { checkNoLeaves<P2A3>(3, 5, 27, 0); }
TEST(merkle_noleaves, tree_min_a4) { checkNoLeaves<P2A4>(4, 5, 64, 0); }
