// The cyclic gsum correction (std_sum.pil): g_r = prefix_r - (r+1)·S/N, S = prefix_{N-1}. The CPU
// (cyclicSumCorrection, hints.cpp) and the GPU (cyclicSumCorrectionGPU, hints.cu) against an exact
// reference, so the two witnesses are the same field elements.
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "hints.cuh"
#include "cuda_utils.cuh"

namespace {

constexpr uint64_t kP = 0xFFFFFFFF00000001ull;

uint64_t addp(uint64_t a, uint64_t b) { return (uint64_t)(((unsigned __int128)a + b) % kP); }
uint64_t subp(uint64_t a, uint64_t b) { return addp(a, kP - b); }
uint64_t mulp(uint64_t a, uint64_t b) { return (uint64_t)(((unsigned __int128)a * b) % kP); }
uint64_t powp(uint64_t a, uint64_t e) {
    uint64_t r = 1;
    for (; e; e >>= 1, a = mulp(a, a)) if (e & 1) r = mulp(r, a);
    return r;
}

// The running sums of random terms (some non-canonical), rows of `dim` limbs.
std::vector<uint64_t> prefix(uint64_t n, uint32_t dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> v(n * dim);
    for (uint64_t i = 0; i < n * dim; ++i) v[i] = rng() % 4 == 0 ? kP + (rng() % 0xFFFFFFFFull) : rng() % kP;
    for (uint64_t i = 1; i < n; ++i)
        for (uint32_t d = 0; d < dim; ++d) v[i * dim + d] = addp(v[i * dim + d], v[(i - 1) * dim + d]);
    return v;
}

std::vector<uint64_t> reference(const std::vector<uint64_t> &v, uint64_t n, uint32_t dim) {
    const uint64_t nInv = powp(n % kP, kP - 2);
    std::vector<uint64_t> r(n * dim);
    for (uint32_t d = 0; d < dim; ++d) {
        const uint64_t k = mulp(v[(n - 1) * dim + d] % kP, nInv);
        for (uint64_t i = 0; i < n; ++i) r[i * dim + d] = subp(v[i * dim + d] % kP, mulp((i + 1) % kP, k));
    }
    return r;
}

std::vector<uint64_t> canonical(const std::vector<uint64_t> &v) {
    std::vector<uint64_t> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = v[i] % kP;
    return r;
}

class CyclicSum : public ::testing::TestWithParam<std::tuple<uint64_t, uint32_t>> {};

TEST_P(CyclicSum, CpuAndGpuMatchTheReference) {
    const auto [n, dim] = GetParam();
    const std::vector<uint64_t> in = prefix(n, dim, n * 31 + dim);
    const std::vector<uint64_t> want = reference(in, n, dim);

    // The CPU indexes rows by FIELD_EXTENSION for dim 3 and by 1 for dim 1: `dim` either way.
    std::vector<uint64_t> cpu = in;
    cyclicSumCorrection((Goldilocks::Element *)cpu.data(), n, dim);

    gl64_t *d = nullptr;
    CHECKCUDAERR(cudaMalloc(&d, in.size() * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(d, in.data(), in.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    cyclicSumCorrectionGPU(d, n, dim, 0);
    std::vector<uint64_t> gpu(in.size());
    CHECKCUDAERR(cudaMemcpy(gpu.data(), d, in.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaFree(d));

    EXPECT_EQ(canonical(cpu), want);
    EXPECT_EQ(canonical(gpu), want);
    for (uint32_t c = 0; c < dim; ++c) {
        EXPECT_EQ(cpu[(n - 1) * dim + c] % kP, 0u) << "cpu last row";
        EXPECT_EQ(gpu[(n - 1) * dim + c] % kP, 0u) << "gpu last row";
    }
    // Cyclic: g_r - g_{r-1} is the term minus S/N on every row, the wrap included.
    const uint64_t nInv = powp(n % kP, kP - 2);
    for (uint32_t c = 0; c < dim; ++c) {
        const uint64_t k = mulp(in[(n - 1) * dim + c] % kP, nInv);
        for (uint64_t i = 0; i < n; ++i) {
            const uint64_t prev = want[((i + n - 1) % n) * dim + c];
            const uint64_t term = i == 0 ? in[c] % kP : subp(in[i * dim + c] % kP, in[(i - 1) * dim + c] % kP);
            ASSERT_EQ(subp(want[i * dim + c], prev), subp(term, k)) << "row " << i;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Sizes, CyclicSum,
                         ::testing::Combine(::testing::Values(2ull, 256ull, 1ull << 12, 1ull << 16),
                                            ::testing::Values(1u, 3u)));

}  // namespace
