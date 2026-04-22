#include "../src/platform.hpp"

#include <gtest/gtest.h>

#if PIL2_HAS_METAL

#include "../src/metal/metal_context.hpp"
#include "../src/goldilocks_base_field.hpp"
#include "../src/ntt_goldilocks.hpp"

#include <cstdint>
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

// Matches the r_[i] = shift^i / N convention used by NTT_Goldilocks::computeR.
std::vector<uint64_t> build_r_table(uint64_t N) {
    std::vector<uint64_t> r_(N);
    Goldilocks::Element shift = Goldilocks::shift();
    Goldilocks::Element inv_N = Goldilocks::inv(Goldilocks::fromU64(N));
    Goldilocks::Element acc   = inv_N;           // r_[0] = 1 * inv_N
    r_[0] = Goldilocks::toU64(acc);
    for (uint64_t i = 1; i < N; ++i) {
        acc   = acc * shift;                     // r_[i] = r_[i-1] * shift = shift^i * inv_N
        r_[i] = Goldilocks::toU64(acc);
    }
    return r_;
}

} // namespace

TEST(MetalLDE, BitExactVsScalarN1024_Ext4096) {
    // N=1024 → N_Extended=4096 (4x extension). Exercises the full LDE
    // pipeline: INTT-with-coset on the N subbuffer, zero-pad, forward NTT
    // on the full N_Extended buffer. Compares byte-for-byte against
    // NTT_Goldilocks::LDE.
    constexpr uint64_t LOG2_N     = 10;
    constexpr uint64_t LOG2_N_EXT = 12;
    constexpr uint64_t N          = 1ULL << LOG2_N;
    constexpr uint64_t N_Ext      = 1ULL << LOG2_N_EXT;

    std::mt19937_64 rng(0xC6057E00ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> input_u64(N);
    for (auto& x : input_u64) x = dist(rng);

    // CPU reference
    std::vector<Goldilocks::Element> cpu_in(N), cpu_out(N_Ext);
    for (uint64_t i = 0; i < N; ++i) cpu_in[i] = Goldilocks::fromU64(input_u64[i]);
    // NTT_Goldilocks::LDE expects the output buffer sized N_Extended; the
    // upper (N_Extended - N) region must be zero before the internal forward
    // NTT. Ensure this by default-initialising cpu_out above.
    NTT_Goldilocks ntt(N);
    ntt.LDE(cpu_out.data(), cpu_in.data(), N_Ext, N, 1);

    std::vector<uint64_t> cpu_u64(N_Ext);
    for (uint64_t i = 0; i < N_Ext; ++i) cpu_u64[i] = Goldilocks::toU64(cpu_out[i]);

    // GPU
    std::vector<uint64_t> gpu_out(N_Ext, 0);
    std::vector<uint64_t> roots = build_roots_table(LOG2_N_EXT);
    std::vector<uint64_t> r_    = build_r_table(N);

    pil2::metal::lde_metal(pil2::metal::get_context(),
                           gpu_out.data(),
                           input_u64.data(),
                           N_Ext, N, 1,
                           roots.data(), N_Ext,
                           r_.data());

    for (uint64_t i = 0; i < N_Ext; ++i) {
        ASSERT_EQ(gpu_out[i], cpu_u64[i])
            << "LDE mismatch at i=" << i
            << " gpu=0x" << std::hex << gpu_out[i]
            << " cpu=0x" << cpu_u64[i];
    }
}

TEST(MetalLDE, BitExactSmallN8_Ext16) {
    // Minimum fused-range size (both N and N_Extended use radix-2 tails
    // since the fused s1s2s3 kernel isn't used by lde_metal's forward
    // path). Validates orchestration on short domains.
    constexpr uint64_t LOG2_N     = 3;
    constexpr uint64_t LOG2_N_EXT = 4;
    constexpr uint64_t N          = 1ULL << LOG2_N;
    constexpr uint64_t N_Ext      = 1ULL << LOG2_N_EXT;

    std::vector<uint64_t> input_u64(N);
    for (uint64_t i = 0; i < N; ++i) input_u64[i] = (i + 1) * 11ULL;

    std::vector<Goldilocks::Element> cpu_in(N), cpu_out(N_Ext);
    for (uint64_t i = 0; i < N; ++i) cpu_in[i] = Goldilocks::fromU64(input_u64[i]);
    NTT_Goldilocks ntt(N);
    ntt.LDE(cpu_out.data(), cpu_in.data(), N_Ext, N, 1);

    std::vector<uint64_t> cpu_u64(N_Ext);
    for (uint64_t i = 0; i < N_Ext; ++i) cpu_u64[i] = Goldilocks::toU64(cpu_out[i]);

    std::vector<uint64_t> gpu_out(N_Ext, 0);
    std::vector<uint64_t> roots = build_roots_table(LOG2_N_EXT);
    std::vector<uint64_t> r_    = build_r_table(N);

    pil2::metal::lde_metal(pil2::metal::get_context(),
                           gpu_out.data(),
                           input_u64.data(),
                           N_Ext, N, 1,
                           roots.data(), N_Ext,
                           r_.data());

    for (uint64_t i = 0; i < N_Ext; ++i) {
        ASSERT_EQ(gpu_out[i], cpu_u64[i]) << "small-LDE mismatch at i=" << i;
    }
}

TEST(MetalLDE, BitExactMultiColN256_Ext1024_Ncols5) {
    // Validates that ncols > 1 works bit-exact against NTT_Goldilocks::LDE.
    // ncols=5 is deliberately non-power-of-two to stress the row-major
    // stride handling inside the Metal kernels. N=256, N_Ext=1024 keeps it
    // quick under CI.
    constexpr uint64_t LOG2_N     = 8;
    constexpr uint64_t LOG2_N_EXT = 10;
    constexpr uint64_t N          = 1ULL << LOG2_N;
    constexpr uint64_t N_Ext      = 1ULL << LOG2_N_EXT;
    constexpr uint64_t NCOLS      = 5;

    std::mt19937_64 rng(0xDEADBEEFULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> input_u64(N * NCOLS);
    for (auto& x : input_u64) x = dist(rng);

    std::vector<Goldilocks::Element> cpu_in(N * NCOLS), cpu_out(N_Ext * NCOLS);
    for (uint64_t i = 0; i < N * NCOLS; ++i) cpu_in[i] = Goldilocks::fromU64(input_u64[i]);
    NTT_Goldilocks ntt(N);
    ntt.LDE(cpu_out.data(), cpu_in.data(), N_Ext, N, NCOLS);

    std::vector<uint64_t> cpu_u64(N_Ext * NCOLS);
    for (uint64_t i = 0; i < N_Ext * NCOLS; ++i) cpu_u64[i] = Goldilocks::toU64(cpu_out[i]);

    std::vector<uint64_t> gpu_out(N_Ext * NCOLS, 0);
    std::vector<uint64_t> roots = build_roots_table(LOG2_N_EXT);
    std::vector<uint64_t> r_    = build_r_table(N);

    pil2::metal::lde_metal(pil2::metal::get_context(),
                           gpu_out.data(),
                           input_u64.data(),
                           N_Ext, N, NCOLS,
                           roots.data(), N_Ext,
                           r_.data());

    for (uint64_t i = 0; i < N_Ext * NCOLS; ++i) {
        ASSERT_EQ(gpu_out[i], cpu_u64[i])
            << "multicol-LDE mismatch at i=" << i
            << " row=" << (i / NCOLS) << " col=" << (i % NCOLS)
            << " gpu=0x" << std::hex << gpu_out[i]
            << " cpu=0x" << cpu_u64[i];
    }
}

#else

TEST(MetalLDE, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif // PIL2_HAS_METAL
