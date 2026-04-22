#include "../src/platform.hpp"

#include <gtest/gtest.h>

#if PIL2_HAS_METAL

// Use the C API exclusively — no direct pil2::metal::* references. This
// mirrors how a Rust FFI caller or a pure-C integration would invoke the
// Metal backend.
#include "../src/metal/metal_c_api.h"
#include "../src/goldilocks_base_field.hpp"
#include "../src/ntt_goldilocks.hpp"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr uint64_t GL_P = 0xFFFFFFFF00000001ULL;

std::vector<uint64_t> build_roots_table(uint64_t log2_N) {
    const uint64_t N = 1ULL << log2_N;
    std::vector<uint64_t> roots(N);
    Goldilocks::Element w   = Goldilocks::w(log2_N);
    Goldilocks::Element acc = Goldilocks::one();
    for (uint64_t k = 0; k < N; ++k) {
        roots[k] = Goldilocks::toU64(acc);
        acc = acc * w;
    }
    return roots;
}

} // namespace

TEST(MetalCApi, AvailableAndNameLookup) {
    ASSERT_EQ(pil2_metal_available(), 1);
    char name[64] = {0};
    size_t n = pil2_metal_device_name(name, sizeof(name));
    EXPECT_GT(n, 0u);
    EXPECT_EQ(std::strlen(name), n);
    std::cout << "[ C-API    ] device: " << name << std::endl;
}

TEST(MetalCApi, NttForwardBitExact) {
    // Full round-trip through the C API: build inputs, call the extern "C"
    // entry, compare to the scalar CPU reference. Validates that the ABI
    // (u64 arrays, size params, roots pointer) is consistent with how the
    // C++ entry works.
    constexpr uint64_t LOG2_N = 10;
    constexpr uint64_t N      = 1ULL << LOG2_N;

    std::mt19937_64 rng(0xC9C0AB1ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);
    std::vector<uint64_t> input(N);
    for (auto& x : input) x = dist(rng);

    std::vector<Goldilocks::Element> cpu_buf(N);
    for (uint64_t i = 0; i < N; ++i) cpu_buf[i] = Goldilocks::fromU64(input[i]);
    NTT_Goldilocks ntt(N);
    ntt.NTT(cpu_buf.data(), cpu_buf.data(), N);

    std::vector<uint64_t> cpu_out(N);
    for (uint64_t i = 0; i < N; ++i) cpu_out[i] = Goldilocks::toU64(cpu_buf[i]);

    std::vector<uint64_t> roots   = build_roots_table(LOG2_N);
    std::vector<uint64_t> gpu_buf = input;
    ASSERT_EQ(pil2_metal_ntt_forward(gpu_buf.data(), N, 1, roots.data(), N), 0);

    for (uint64_t i = 0; i < N; ++i) {
        ASSERT_EQ(gpu_buf[i], cpu_out[i]) << "C-API NTT mismatch at i=" << i;
    }
}

#else

TEST(MetalCApi, UnavailableReturnsZero) {
    EXPECT_EQ(pil2_metal_available(), 0);
}

#endif // PIL2_HAS_METAL
