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

struct Gl3 { uint64_t c0, c1, c2; };

// Build CPU reference output by calling Goldilocks3::{add,sub,mul}.
// Works element-by-element; inputs and outputs are raw canonical u64s.
std::vector<uint64_t> cpu_gl3_op(const std::string& op,
                                 const std::vector<uint64_t>& a,
                                 const std::vector<uint64_t>& b,
                                 size_t n) {
    std::vector<uint64_t> out(n * 3);
    for (size_t i = 0; i < n; ++i) {
        Goldilocks3::Element ae, be, re;
        for (int c = 0; c < 3; ++c) {
            ae[c] = Goldilocks::fromU64(a[i*3+c]);
            if (op != "mul_scalar") be[c] = Goldilocks::fromU64(b[i*3+c]);
        }
        if (op == "add")        Goldilocks3::add(re, ae, be);
        else if (op == "sub")   Goldilocks3::sub(re, ae, be);
        else if (op == "mul")   Goldilocks3::mul(re, ae, be);
        else if (op == "mul_scalar") {
            Goldilocks::Element s = Goldilocks::fromU64(b[i]);
            Goldilocks3::mul(re, ae, s);
        } else { ADD_FAILURE() << "unknown op " << op; }
        for (int c = 0; c < 3; ++c) out[i*3+c] = Goldilocks::toU64(re[c]);
    }
    return out;
}

void check_op(const std::string& op, size_t n, uint64_t seed,
              bool b_is_scalar = false) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> a(n * 3);
    std::vector<uint64_t> b(b_is_scalar ? n : n * 3);
    for (auto& x : a) x = dist(rng);
    for (auto& x : b) x = dist(rng);

    auto cpu = cpu_gl3_op(op, a, b, n);

    std::vector<uint64_t> gpu(n * 3, 0);
    pil2::metal::run_gl3_op(pil2::metal::get_context(),
                            op.c_str(), a.data(), b.data(),
                            gpu.data(), static_cast<uint32_t>(n));

    for (size_t i = 0; i < n * 3; ++i) {
        ASSERT_EQ(gpu[i], cpu[i])
            << op << " mismatch at i=" << i
            << " (row " << (i/3) << " col " << (i%3) << ")"
            << " gpu=0x" << std::hex << gpu[i]
            << " cpu=0x" << cpu[i];
    }
}

} // namespace

TEST(MetalGl3, AddRandomN1024)       { check_op("add", 1024, 0xA1D0ULL); }
TEST(MetalGl3, SubRandomN1024)       { check_op("sub", 1024, 0x5B0ULL);  }
TEST(MetalGl3, MulRandomN1024)       { check_op("mul", 1024, 0xC0FFEEULL);  }
TEST(MetalGl3, MulScalarRandomN1024) { check_op("mul_scalar", 1024, 0xA1FA1FULL, /*b_is_scalar=*/true); }

// run_gl3_mul — production wrapper that uses PSO cache + resolver-based
// zero-copy. Exercises both paths so a regression in either one surfaces
// here before reaching the expression-VM bridge.

// Scratch-fallback path: all pointers live on the Rust heap (testing
// framework std::vector), so metal_resolve_shared misses on every one
// and the scratch pool + memcpy path runs.
TEST(MetalGl3Mul, HeapHeapHeap_N1024) {
    constexpr size_t n = 1024;
    std::mt19937_64 rng(0xC0FFEE11ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);
    std::vector<uint64_t> a(n*3), b(n*3), gpu(n*3, 0);
    for (auto& x : a) x = dist(rng);
    for (auto& x : b) x = dist(rng);
    auto cpu = cpu_gl3_op("mul", a, b, n);
    pil2::metal::run_gl3_mul(pil2::metal::get_context(),
                             a.data(), b.data(), gpu.data(),
                             static_cast<uint32_t>(n));
    for (size_t i = 0; i < n*3; ++i) {
        ASSERT_EQ(gpu[i], cpu[i])
            << "gl3_mul heap mismatch at i=" << i
            << " gpu=0x" << std::hex << gpu[i] << " cpu=0x" << cpu[i];
    }
}

// Resolver-hit path: all three buffers allocated via metal_alloc_shared,
// so every metal_resolve_shared succeeds and the kernel binds directly
// (no memcpy, no scratch borrow). Same inputs as above for easy diff.
TEST(MetalGl3Mul, SharedSharedShared_N1024) {
    constexpr size_t n = 1024;
    std::mt19937_64 rng(0xC0FFEE22ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);
    const size_t bytes = n * 3 * sizeof(uint64_t);
    uint64_t* a = static_cast<uint64_t*>(pil2::metal::metal_alloc_shared(bytes));
    uint64_t* b = static_cast<uint64_t*>(pil2::metal::metal_alloc_shared(bytes));
    uint64_t* gpu = static_cast<uint64_t*>(pil2::metal::metal_alloc_shared(bytes));
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(gpu, nullptr);
    for (size_t i = 0; i < n*3; ++i) { a[i] = dist(rng); b[i] = dist(rng); gpu[i] = 0; }
    std::vector<uint64_t> a_vec(a, a + n*3), b_vec(b, b + n*3);
    auto cpu = cpu_gl3_op("mul", a_vec, b_vec, n);
    pil2::metal::run_gl3_mul(pil2::metal::get_context(),
                             a, b, gpu, static_cast<uint32_t>(n));
    for (size_t i = 0; i < n*3; ++i) {
        ASSERT_EQ(gpu[i], cpu[i])
            << "gl3_mul shared mismatch at i=" << i
            << " gpu=0x" << std::hex << gpu[i] << " cpu=0x" << cpu[i];
    }
    pil2::metal::metal_free_shared(a);
    pil2::metal::metal_free_shared(b);
    pil2::metal::metal_free_shared(gpu);
}

// Strided dst — shared-allocation path (resolver hits). dst_stride 12
// simulates a 3-col-cubic imPol being written into a 12-col cm-section
// row. Between-row cells (4..11) must NOT be touched by the kernel; we
// seed them with a sentinel and re-check at the end.
TEST(MetalGl3Mul, SharedDstStrided12_N256) {
    constexpr size_t n = 256;
    constexpr uint32_t stride = 12;
    constexpr uint64_t GAP_SENTINEL = 0xDEADBEEFCAFEBABEULL;
    std::mt19937_64 rng(0xABBA5510ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    const size_t src_bytes = n * 3 * sizeof(uint64_t);
    // Round dst alloc up past the last written cell so the sentinel at
    // the tail is real memory we can read back.
    const size_t dst_alloc_bytes = n * stride * sizeof(uint64_t);

    uint64_t* a   = static_cast<uint64_t*>(pil2::metal::metal_alloc_shared(src_bytes));
    uint64_t* b   = static_cast<uint64_t*>(pil2::metal::metal_alloc_shared(src_bytes));
    uint64_t* gpu = static_cast<uint64_t*>(pil2::metal::metal_alloc_shared(dst_alloc_bytes));
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(gpu, nullptr);

    for (size_t i = 0; i < n*3; ++i) { a[i] = dist(rng); b[i] = dist(rng); }
    for (size_t i = 0; i < n*stride; ++i) gpu[i] = GAP_SENTINEL;

    std::vector<uint64_t> a_vec(a, a + n*3), b_vec(b, b + n*3);
    auto cpu = cpu_gl3_op("mul", a_vec, b_vec, n);
    pil2::metal::run_gl3_mul(pil2::metal::get_context(),
                             a, b, gpu, static_cast<uint32_t>(n), stride);

    // Check the 3 written cells per row + that the gap cells stayed sentinel.
    for (size_t t = 0; t < n; ++t) {
        const size_t base = t * stride;
        for (int c = 0; c < 3; ++c) {
            ASSERT_EQ(gpu[base + c], cpu[t*3 + c])
                << "written cell mismatch t=" << t << " c=" << c
                << " gpu=0x" << std::hex << gpu[base + c]
                << " cpu=0x" << cpu[t*3 + c];
        }
        for (uint32_t c = 3; c < stride; ++c) {
            ASSERT_EQ(gpu[base + c], GAP_SENTINEL)
                << "gap cell clobbered t=" << t << " c=" << c
                << " val=0x" << std::hex << gpu[base + c];
        }
    }
    pil2::metal::metal_free_shared(a);
    pil2::metal::metal_free_shared(b);
    pil2::metal::metal_free_shared(gpu);
}

// Strided dst — heap (resolver miss) fallback path. Same invariant: the
// scratch-based copy-back must NOT clobber the gap cells in `out`.
TEST(MetalGl3Mul, HeapDstStrided55_N128) {
    constexpr size_t n = 128;
    constexpr uint32_t stride = 55;
    constexpr uint64_t GAP_SENTINEL = 0x1122334455667788ULL;
    std::mt19937_64 rng(0xFADE5511ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> a(n*3), b(n*3);
    std::vector<uint64_t> gpu(n*stride, GAP_SENTINEL);
    for (auto& x : a) x = dist(rng);
    for (auto& x : b) x = dist(rng);

    auto cpu = cpu_gl3_op("mul", a, b, n);
    pil2::metal::run_gl3_mul(pil2::metal::get_context(),
                             a.data(), b.data(), gpu.data(),
                             static_cast<uint32_t>(n), stride);

    for (size_t t = 0; t < n; ++t) {
        const size_t base = t * stride;
        for (int c = 0; c < 3; ++c) {
            ASSERT_EQ(gpu[base + c], cpu[t*3 + c])
                << "written cell mismatch t=" << t << " c=" << c;
        }
        for (uint32_t c = 3; c < stride; ++c) {
            ASSERT_EQ(gpu[base + c], GAP_SENTINEL)
                << "gap cell clobbered t=" << t << " c=" << c;
        }
    }
}

// Edge case: n=1 — smallest valid dispatch. Catches any dispatchThreads
// math that assumed non-trivial grid size.
TEST(MetalGl3Mul, SingleElement) {
    const uint64_t a[3] = {1, 2, 3};
    const uint64_t b[3] = {4, 5, 6};
    uint64_t gpu[3] = {0, 0, 0};
    std::vector<uint64_t> a_v(a, a+3), b_v(b, b+3);
    auto cpu = cpu_gl3_op("mul", a_v, b_v, 1);
    pil2::metal::run_gl3_mul(pil2::metal::get_context(),
                             a, b, gpu, 1);
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ(gpu[i], cpu[i])
            << "gl3_mul n=1 mismatch at c=" << i
            << " gpu=0x" << std::hex << gpu[i] << " cpu=0x" << cpu[i];
    }
}

// Edge cases: 0, 1, -1 ≡ p-1, and the carry-provoking pattern from the
// gl_add bug fix (a=1, b=P-1 in one limb) to confirm gl3_add inherits
// the canonicalisation.
TEST(MetalGl3, EdgeCases) {
    const uint64_t P = GL_P;
    const std::vector<Gl3> inputs_a = {
        {0, 0, 0},
        {1, 0, 0},
        {P - 1, P - 1, P - 1},
        {1, P - 1, 0},          // a+b == P when b is {P-1, 1, 0}
        {P/2, 7, P/3},
    };
    const std::vector<Gl3> inputs_b = {
        {0, 0, 0},
        {0, 0, 0},
        {1, 1, 1},
        {P - 1, 1, 0},
        {13, P - 7, 42},
    };
    std::vector<uint64_t> a_flat, b_flat;
    for (size_t i = 0; i < inputs_a.size(); ++i) {
        a_flat.push_back(inputs_a[i].c0);
        a_flat.push_back(inputs_a[i].c1);
        a_flat.push_back(inputs_a[i].c2);
        b_flat.push_back(inputs_b[i].c0);
        b_flat.push_back(inputs_b[i].c1);
        b_flat.push_back(inputs_b[i].c2);
    }
    const size_t n = inputs_a.size();
    for (const char* op : { "add", "sub", "mul" }) {
        auto cpu = cpu_gl3_op(op, a_flat, b_flat, n);
        std::vector<uint64_t> gpu(n * 3, 0);
        pil2::metal::run_gl3_op(pil2::metal::get_context(),
                                op, a_flat.data(), b_flat.data(),
                                gpu.data(), static_cast<uint32_t>(n));
        for (size_t i = 0; i < n * 3; ++i) {
            ASSERT_EQ(gpu[i], cpu[i])
                << op << " edge mismatch at i=" << i
                << " gpu=0x" << std::hex << gpu[i]
                << " cpu=0x" << cpu[i];
        }
    }
}

#else

TEST(MetalGl3, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif // PIL2_HAS_METAL
