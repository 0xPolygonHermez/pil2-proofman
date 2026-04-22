#include "../src/platform.hpp"

#include <gtest/gtest.h>

#if PIL2_HAS_METAL

#include "../src/metal/metal_context.hpp"
#include "../src/goldilocks_base_field.hpp"
#include "../src/goldilocks_cubic_extension.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr uint64_t GL_P = 0xFFFFFFFF00000001ULL;

} // namespace

// Base-field Fermat inverse bit-exact with Goldilocks::inv for a sweep of
// random non-zero inputs plus hand-picked edge cases (1, p-1, small
// numbers, largest valid). Also verifies the algebraic property
// a * inv(a) == 1 mod p via the GPU result alone.
TEST(MetalGlInv, BasicCases) {
    std::vector<uint64_t> in = { 1ULL, 2ULL, 3ULL, GL_P - 1ULL, GL_P - 2ULL,
                                  0x100000000ULL, 0x1FFFFFFFFULL,
                                  0x123456789ABCDEFULL };
    std::mt19937_64 rng(0xB11A0001ULL);
    std::uniform_int_distribution<uint64_t> dist(1, GL_P - 1);
    for (int i = 0; i < 64; ++i) in.push_back(dist(rng));

    const uint32_t n = (uint32_t)in.size();
    std::vector<uint64_t> gpu(n, 0);
    pil2::metal::run_gl_inv_test(pil2::metal::get_context(),
                                 in.data(), gpu.data(), n);

    for (uint32_t i = 0; i < n; ++i) {
        Goldilocks::Element e = Goldilocks::fromU64(in[i]);
        Goldilocks::Element expected;
        Goldilocks::inv(expected, e);
        uint64_t want = Goldilocks::toU64(expected);
        ASSERT_EQ(gpu[i], want)
            << "i=" << i << " in=0x" << std::hex << in[i]
            << " got=0x" << gpu[i] << " want=0x" << want;

        // Algebraic check as a second safety net: a * inv(a) == 1 via
        // the GPU-produced inverse only.
        Goldilocks::Element g = Goldilocks::fromU64(gpu[i]);
        Goldilocks::Element product = e * g;
        ASSERT_EQ(Goldilocks::toU64(product), 1ULL)
            << "i=" << i << " a * gpu_inv != 1";
    }
}

// Cubic-extension inverse bit-exact with Goldilocks3::inv. Exercises
// both hand-picked edge elements and random cubic inputs. Also verifies
// a * inv(a) == (1, 0, 0) via the GPU output alone.
TEST(MetalGl3Inv, BasicCases) {
    std::vector<std::array<uint64_t, 3>> in = {
        {1ULL, 0ULL, 0ULL},            // identity
        {0ULL, 1ULL, 0ULL},            // x
        {0ULL, 0ULL, 1ULL},            // x^2
        {2ULL, 3ULL, 5ULL},
        {GL_P - 1, GL_P - 1, GL_P - 1},
        {GL_P - 1, 1ULL,   2ULL},
    };
    std::mt19937_64 rng(0xB11A0003ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);
    for (int i = 0; i < 32; ++i) {
        // Reject-sample the zero cubic element; the iteration is
        // guaranteed to terminate — zero is a measure-zero event among
        // uniform triples over a prime field.
        std::array<uint64_t, 3> t;
        do {
            t = { dist(rng), dist(rng), dist(rng) };
        } while (t[0] == 0 && t[1] == 0 && t[2] == 0);
        in.push_back(t);
    }

    const uint32_t n = (uint32_t)in.size();
    std::vector<uint64_t> a_flat(n * 3);
    for (uint32_t i = 0; i < n; ++i) {
        a_flat[i * 3 + 0] = in[i][0];
        a_flat[i * 3 + 1] = in[i][1];
        a_flat[i * 3 + 2] = in[i][2];
    }

    std::vector<uint64_t> gpu(n * 3, 0);
    pil2::metal::run_gl3_inv_test(pil2::metal::get_context(),
                                  a_flat.data(), gpu.data(), n);

    for (uint32_t i = 0; i < n; ++i) {
        Goldilocks3::Element e;
        for (int c = 0; c < 3; ++c) e[c] = Goldilocks::fromU64(in[i][c]);
        Goldilocks3::Element expected;
        Goldilocks3::inv(expected, e);
        for (int c = 0; c < 3; ++c) {
            uint64_t want = Goldilocks::toU64(expected[c]);
            ASSERT_EQ(gpu[i * 3 + c], want)
                << "i=" << i << " c=" << c
                << " got=0x" << std::hex << gpu[i * 3 + c]
                << " want=0x" << want;
        }

        // a * inv(a) == (1, 0, 0).
        Goldilocks3::Element gpu_inv;
        for (int c = 0; c < 3; ++c) gpu_inv[c] = Goldilocks::fromU64(gpu[i * 3 + c]);
        Goldilocks3::Element product;
        Goldilocks3::mul(product, e, gpu_inv);
        ASSERT_EQ(Goldilocks::toU64(product[0]), 1ULL) << "i=" << i << " c0";
        ASSERT_EQ(Goldilocks::toU64(product[1]), 0ULL) << "i=" << i << " c1";
        ASSERT_EQ(Goldilocks::toU64(product[2]), 0ULL) << "i=" << i << " c2";
    }
}

#else

TEST(MetalGlInv, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif // PIL2_HAS_METAL
