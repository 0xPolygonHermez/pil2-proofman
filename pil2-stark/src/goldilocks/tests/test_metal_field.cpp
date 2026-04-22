#include "../src/platform.hpp"

#include <gtest/gtest.h>

#if PIL2_HAS_METAL

#include "../src/metal/metal_context.hpp"
#include "../src/goldilocks_base_field.hpp"

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t GL_P = 0xFFFFFFFF00000001ULL;

// Fills `v` with uniform-random u64 values in [0, p). fromU64 on the CPU
// reference side accepts raw u64 but gl_add/sub/mul on the GPU side assume
// canonical inputs, so we only test canonical → canonical behaviour here.
void fill_uniform(std::vector<uint64_t>& v, std::mt19937_64& rng) {
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);
    for (auto& x : v) x = dist(rng);
}

// Overwrites the first few entries of a/b with edge cases that exercise the
// add/sub carry paths and the mul high-limb corrections. All pairs are
// canonical; the test comparison is against the fully-reduced CPU output.
void inject_edge_cases(std::vector<uint64_t>& a, std::vector<uint64_t>& b) {
    const std::pair<uint64_t, uint64_t> cases[] = {
        {0, 0},
        {0, GL_P - 1},
        {GL_P - 1, 0},
        {GL_P - 1, GL_P - 1},
        {1, GL_P - 1},
        {GL_P - 1, 1},
        {GL_P / 2, GL_P / 2},
        {0xFFFFFFFFULL,           0xFFFFFFFF00000000ULL},  // add double-carry
        {0x8000000000000000ULL,   0x8000000000000000ULL},  // lazy mul path
        {0xFFFFFFFE00000001ULL,   0xFFFFFFFE00000001ULL},  // mul near-prime
    };
    size_t i = 0;
    for (const auto& [x, y] : cases) {
        if (i >= a.size()) break;
        a[i] = x;
        b[i] = y;
        ++i;
    }
}

struct FieldTestCase {
    const char* op;
    uint64_t (*cpu_fn)(uint64_t, uint64_t);
};

uint64_t cpu_add(uint64_t a, uint64_t b) {
    return Goldilocks::toU64(Goldilocks::add(Goldilocks::fromU64(a),
                                             Goldilocks::fromU64(b)));
}
uint64_t cpu_sub(uint64_t a, uint64_t b) {
    return Goldilocks::toU64(Goldilocks::sub(Goldilocks::fromU64(a),
                                             Goldilocks::fromU64(b)));
}
uint64_t cpu_mul(uint64_t a, uint64_t b) {
    return Goldilocks::toU64(Goldilocks::mul(Goldilocks::fromU64(a),
                                             Goldilocks::fromU64(b)));
}

void run_op_bit_exact(const FieldTestCase& tc, uint64_t seed) {
    constexpr uint32_t N = 4096;
    std::mt19937_64 rng(seed);

    std::vector<uint64_t> a(N), b(N), gpu_out(N, 0), cpu_out(N);
    fill_uniform(a, rng);
    fill_uniform(b, rng);
    inject_edge_cases(a, b);

    for (uint32_t i = 0; i < N; ++i) {
        cpu_out[i] = tc.cpu_fn(a[i], b[i]);
    }

    auto* ctx = pil2::metal::get_context();
    pil2::metal::run_field_op(ctx, tc.op, a.data(), b.data(), gpu_out.data(), N);

    for (uint32_t i = 0; i < N; ++i) {
        ASSERT_EQ(gpu_out[i], cpu_out[i])
            << "op=" << tc.op
            << " i=" << i
            << " a=0x" << std::hex << a[i]
            << " b=0x" << b[i]
            << " gpu=0x" << gpu_out[i]
            << " cpu=0x" << cpu_out[i];
    }
}

} // anonymous namespace

TEST(MetalField, AddBitExact) {
    run_op_bit_exact({"add", cpu_add}, 0xC400ADDULL);
}
TEST(MetalField, SubBitExact) {
    run_op_bit_exact({"sub", cpu_sub}, 0xC400B5BULL);
}
TEST(MetalField, MulBitExact) {
    run_op_bit_exact({"mul", cpu_mul}, 0xC400E42ULL);
}

#else

TEST(MetalField, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif // PIL2_HAS_METAL
