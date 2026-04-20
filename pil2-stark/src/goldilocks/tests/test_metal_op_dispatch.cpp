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

// CPU reference for all three op flavors. Matches Goldilocks::op_pack /
// Goldilocks3::op_pack / Goldilocks3::op_31_pack element-by-element.
// Not perf-optimised — used as bit-exact oracle.

void cpu_gl_op_ref(uint32_t op, const std::vector<uint64_t>& a,
                   const std::vector<uint64_t>& b, std::vector<uint64_t>& out) {
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        Goldilocks::Element ae = Goldilocks::fromU64(a[i]);
        Goldilocks::Element be = Goldilocks::fromU64(b[i]);
        Goldilocks::Element re;
        switch (op) {
            case 0: re = ae + be; break;
            case 1: re = ae - be; break;
            case 2: re = ae * be; break;
            default: re = be - ae; break;  // 3
        }
        out[i] = Goldilocks::toU64(re);
    }
}

void cpu_gl3_op_ref(uint32_t op, const std::vector<uint64_t>& a,
                    const std::vector<uint64_t>& b, std::vector<uint64_t>& out) {
    const size_t n = a.size() / 3;
    for (size_t i = 0; i < n; ++i) {
        Goldilocks3::Element ae, be, re;
        for (int c = 0; c < 3; ++c) {
            ae[c] = Goldilocks::fromU64(a[i*3+c]);
            be[c] = Goldilocks::fromU64(b[i*3+c]);
        }
        switch (op) {
            case 0: Goldilocks3::add(re, ae, be); break;
            case 1: Goldilocks3::sub(re, ae, be); break;
            case 2: Goldilocks3::mul(re, ae, be); break;
            default: Goldilocks3::sub(re, be, ae); break;
        }
        for (int c = 0; c < 3; ++c) out[i*3+c] = Goldilocks::toU64(re[c]);
    }
}

void cpu_gl3_op_31_ref(uint32_t op, const std::vector<uint64_t>& a,
                       const std::vector<uint64_t>& b, std::vector<uint64_t>& out) {
    const size_t n = b.size();
    for (size_t i = 0; i < n; ++i) {
        Goldilocks3::Element ae, re;
        for (int c = 0; c < 3; ++c) ae[c] = Goldilocks::fromU64(a[i*3+c]);
        Goldilocks::Element bs = Goldilocks::fromU64(b[i]);
        switch (op) {
            case 0: Goldilocks3::add(re, ae, bs); break;
            case 1: Goldilocks3::sub(re, ae, bs); break;
            case 2: Goldilocks3::mul(re, ae, bs); break;
            default: Goldilocks3::sub(re, bs, ae); break;
        }
        for (int c = 0; c < 3; ++c) out[i*3+c] = Goldilocks::toU64(re[c]);
    }
}

void run_and_check(const std::string& flavor, uint32_t op, size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> a, b, out_cpu, out_gpu;
    if (flavor == "gl_op") {
        a.resize(n); b.resize(n); out_cpu.resize(n); out_gpu.resize(n, 0);
        for (auto& x : a) x = dist(rng);
        for (auto& x : b) x = dist(rng);
        cpu_gl_op_ref(op, a, b, out_cpu);
    } else if (flavor == "gl3_op") {
        a.resize(n*3); b.resize(n*3); out_cpu.resize(n*3); out_gpu.resize(n*3, 0);
        for (auto& x : a) x = dist(rng);
        for (auto& x : b) x = dist(rng);
        cpu_gl3_op_ref(op, a, b, out_cpu);
    } else {  // gl3_op_31
        a.resize(n*3); b.resize(n); out_cpu.resize(n*3); out_gpu.resize(n*3, 0);
        for (auto& x : a) x = dist(rng);
        for (auto& x : b) x = dist(rng);
        cpu_gl3_op_31_ref(op, a, b, out_cpu);
    }

    pil2::metal::run_op_dispatch(pil2::metal::get_context(),
                                 flavor.c_str(), op,
                                 a.data(), b.data(), out_gpu.data(),
                                 static_cast<uint32_t>(n));
    for (size_t i = 0; i < out_cpu.size(); ++i) {
        ASSERT_EQ(out_gpu[i], out_cpu[i])
            << flavor << " op=" << op << " mismatch at i=" << i
            << " gpu=0x" << std::hex << out_gpu[i]
            << " cpu=0x" << out_cpu[i];
    }
}

} // namespace

// Base field: all 4 op codes.
TEST(MetalOpDispatch, GlOpAllCodesN512) {
    for (uint32_t op = 0; op < 4; ++op) {
        SCOPED_TRACE(::testing::Message() << "gl_op op=" << op);
        run_and_check("gl_op", op, 512, 0xA10000ULL + op);
    }
}

// Cubic × cubic: all 4 op codes.
TEST(MetalOpDispatch, Gl3OpAllCodesN256) {
    for (uint32_t op = 0; op < 4; ++op) {
        SCOPED_TRACE(::testing::Message() << "gl3_op op=" << op);
        run_and_check("gl3_op", op, 256, 0xB30000ULL + op);
    }
}

// Cubic × base (op_31): all 4 op codes.
TEST(MetalOpDispatch, Gl3Op31AllCodesN256) {
    for (uint32_t op = 0; op < 4; ++op) {
        SCOPED_TRACE(::testing::Message() << "gl3_op_31 op=" << op);
        run_and_check("gl3_op_31", op, 256, 0xC31000ULL + op);
    }
}

#else

TEST(MetalOpDispatch, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif // PIL2_HAS_METAL
