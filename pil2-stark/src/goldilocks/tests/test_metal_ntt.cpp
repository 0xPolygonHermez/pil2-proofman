#include "../src/platform.hpp"

#include <gtest/gtest.h>

#if PIL2_HAS_METAL

#include "../src/metal/metal_context.hpp"
#include "../src/goldilocks_base_field.hpp"
#include "../src/ntt_goldilocks.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr uint64_t GL_P = 0xFFFFFFFF00000001ULL;

// Build the full primitive (N)-th roots-of-unity table as raw u64. Indexed
// so roots[k] = g^k mod p where g = Goldilocks::w(log2_N). This is what
// NTT_Goldilocks(N) also uses internally (its stored roots[] array has the
// same contract, derived from the same Goldilocks::w(log2_N)).
std::vector<uint64_t> build_roots_table(uint64_t log2_N) {
    const uint64_t N = 1ULL << log2_N;
    std::vector<uint64_t> roots(N);
    Goldilocks::Element w = Goldilocks::w(log2_N);
    Goldilocks::Element acc = Goldilocks::one();
    for (uint64_t k = 0; k < N; ++k) {
        roots[k] = Goldilocks::toU64(acc);
        acc = acc * w;
    }
    return roots;
}

} // namespace

TEST(MetalNTT, ForwardBitExactN1024) {
    constexpr uint64_t LOG2_N = 10;
    constexpr uint64_t N      = 1ULL << LOG2_N;

    std::mt19937_64 rng(0xC5A0F000ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    // Random input in [0, p). Keep a raw u64 copy for the GPU path and a
    // parallel Element copy for the CPU reference.
    std::vector<uint64_t> input_u64(N);
    for (auto& x : input_u64) x = dist(rng);

    std::vector<Goldilocks::Element> cpu_buf(N);
    for (uint64_t i = 0; i < N; ++i) {
        cpu_buf[i] = Goldilocks::fromU64(input_u64[i]);
    }

    NTT_Goldilocks ntt(N);
    ntt.NTT(cpu_buf.data(), cpu_buf.data(), N);  // forward, in-place

    std::vector<uint64_t> cpu_out(N);
    for (uint64_t i = 0; i < N; ++i) {
        cpu_out[i] = Goldilocks::toU64(cpu_buf[i]);
    }

    std::vector<uint64_t> gpu_buf = input_u64;
    std::vector<uint64_t> roots   = build_roots_table(LOG2_N);

    auto* ctx = pil2::metal::get_context();
    pil2::metal::ntt_forward_metal(ctx,
                                   gpu_buf.data(),
                                   /*size*/ N,
                                   /*ncols*/ 1,
                                   roots.data(),
                                   /*roots_len*/ N);

    for (uint64_t i = 0; i < N; ++i) {
        ASSERT_EQ(gpu_buf[i], cpu_out[i])
            << "NTT mismatch at i=" << i
            << " input=0x" << std::hex << input_u64[i]
            << " gpu=0x" << gpu_buf[i]
            << " cpu=0x" << cpu_out[i];
    }
}

TEST(MetalNTT, ForwardBitExactN2048) {
    // log2(N) == 11 exercises the radix-8 + radix-4-tail path: three
    // radix-8 dispatches cover stages 1-9, then a radix-4 tail covers
    // stages 10-11. This is the only phase-loop branch not hit by the
    // other sizes (N=1024 needs a radix-2 tail; N=512 is pure radix-8;
    // N=16 is radix-8 + radix-2 tail).
    constexpr uint64_t LOG2_N = 11;
    constexpr uint64_t N      = 1ULL << LOG2_N;

    std::mt19937_64 rng(0xC5B20D00ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> input_u64(N);
    for (auto& x : input_u64) x = dist(rng);

    std::vector<Goldilocks::Element> cpu_buf(N);
    for (uint64_t i = 0; i < N; ++i) cpu_buf[i] = Goldilocks::fromU64(input_u64[i]);

    NTT_Goldilocks ntt(N);
    ntt.NTT(cpu_buf.data(), cpu_buf.data(), N);

    std::vector<uint64_t> cpu_out(N);
    for (uint64_t i = 0; i < N; ++i) cpu_out[i] = Goldilocks::toU64(cpu_buf[i]);

    std::vector<uint64_t> gpu_buf = input_u64;
    std::vector<uint64_t> roots   = build_roots_table(LOG2_N);

    pil2::metal::ntt_forward_metal(pil2::metal::get_context(),
                                   gpu_buf.data(), N, 1, roots.data(), N);

    for (uint64_t i = 0; i < N; ++i) {
        ASSERT_EQ(gpu_buf[i], cpu_out[i]) << "radix-8+r4-tail mismatch at i=" << i;
    }
}

TEST(MetalNTT, ForwardBitExactOddLog2N512) {
    // Odd log2(N) forces the radix-4 dispatcher to emit one radix-2 tail
    // stage (s == domain_pow, no pair available). Validates that the mixed
    // dispatch path agrees with the scalar reference.
    constexpr uint64_t LOG2_N = 9;
    constexpr uint64_t N      = 1ULL << LOG2_N;

    std::mt19937_64 rng(0xC5B00D00ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> input_u64(N);
    for (auto& x : input_u64) x = dist(rng);

    std::vector<Goldilocks::Element> cpu_buf(N);
    for (uint64_t i = 0; i < N; ++i) cpu_buf[i] = Goldilocks::fromU64(input_u64[i]);

    NTT_Goldilocks ntt(N);
    ntt.NTT(cpu_buf.data(), cpu_buf.data(), N);

    std::vector<uint64_t> cpu_out(N);
    for (uint64_t i = 0; i < N; ++i) cpu_out[i] = Goldilocks::toU64(cpu_buf[i]);

    std::vector<uint64_t> gpu_buf = input_u64;
    std::vector<uint64_t> roots   = build_roots_table(LOG2_N);

    pil2::metal::ntt_forward_metal(pil2::metal::get_context(),
                                   gpu_buf.data(), N, 1, roots.data(), N);

    for (uint64_t i = 0; i < N; ++i) {
        ASSERT_EQ(gpu_buf[i], cpu_out[i]) << "odd-log2(N) mismatch at i=" << i;
    }
}

TEST(MetalNTT, ForwardBitExactMinFusedN8) {
    // N=8 is the smallest domain for which the fused rev+s1s2s3 kernel
    // applies (domain_pow == 3, all three stages consumed by the fused
    // pass, phase loop doesn't execute). Exercises the fused-only path.
    constexpr uint64_t LOG2_N = 3;
    constexpr uint64_t N      = 1ULL << LOG2_N;

    std::vector<Goldilocks::Element> cpu_buf(N);
    std::vector<uint64_t> input_u64(N);
    for (uint64_t i = 0; i < N; ++i) {
        input_u64[i] = (i + 1) * 17ULL;  // mildly spread values
        cpu_buf[i]   = Goldilocks::fromU64(input_u64[i]);
    }

    NTT_Goldilocks ntt(N);
    ntt.NTT(cpu_buf.data(), cpu_buf.data(), N);

    std::vector<uint64_t> cpu_out(N);
    for (uint64_t i = 0; i < N; ++i) cpu_out[i] = Goldilocks::toU64(cpu_buf[i]);

    std::vector<uint64_t> gpu_buf = input_u64;
    std::vector<uint64_t> roots   = build_roots_table(LOG2_N);

    pil2::metal::ntt_forward_metal(pil2::metal::get_context(),
                                   gpu_buf.data(), N, 1, roots.data(), N);

    for (uint64_t i = 0; i < N; ++i) {
        ASSERT_EQ(gpu_buf[i], cpu_out[i]) << "fused-only mismatch at i=" << i;
    }
}

TEST(MetalNTT, ForwardBitExactSmallN16) {
    // Tiny domain exercises the end of the phase loop where m == N and
    // there's exactly one butterfly group. If the per-stage indexing math
    // were wrong, small N would surface it before large N does.
    constexpr uint64_t LOG2_N = 4;
    constexpr uint64_t N      = 1ULL << LOG2_N;

    std::vector<Goldilocks::Element> cpu_buf(N);
    std::vector<uint64_t> input_u64(N);
    for (uint64_t i = 0; i < N; ++i) {
        input_u64[i] = i + 1;  // deterministic, small values
        cpu_buf[i]   = Goldilocks::fromU64(input_u64[i]);
    }

    NTT_Goldilocks ntt(N);
    ntt.NTT(cpu_buf.data(), cpu_buf.data(), N);

    std::vector<uint64_t> cpu_out(N);
    for (uint64_t i = 0; i < N; ++i) cpu_out[i] = Goldilocks::toU64(cpu_buf[i]);

    std::vector<uint64_t> gpu_buf = input_u64;
    std::vector<uint64_t> roots   = build_roots_table(LOG2_N);

    pil2::metal::ntt_forward_metal(pil2::metal::get_context(),
                                   gpu_buf.data(), N, 1,
                                   roots.data(), N);

    for (uint64_t i = 0; i < N; ++i) {
        ASSERT_EQ(gpu_buf[i], cpu_out[i]) << "small-N NTT mismatch at i=" << i;
    }
}

// ---- C5c: INTT (inverse NTT) -----------------------------------------------

namespace {

uint64_t goldilocks_inv_u64(uint64_t n) {
    return Goldilocks::toU64(Goldilocks::inv(Goldilocks::fromU64(n)));
}

} // namespace

TEST(MetalINTT, RoundTripN1024) {
    // INTT(NTT(x)) == x is the cleanest end-to-end correctness gate: it
    // exercises the full forward + inverse Metal pipeline with no scalar
    // CPU intermediate as the reference. Any inconsistency between the
    // forward and inverse paths surfaces as a diff against the original.
    constexpr uint64_t LOG2_N = 10;
    constexpr uint64_t N      = 1ULL << LOG2_N;

    std::mt19937_64 rng(0xC5C0F0DEULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> orig(N);
    for (auto& x : orig) x = dist(rng);

    std::vector<uint64_t> buf   = orig;
    std::vector<uint64_t> roots = build_roots_table(LOG2_N);
    const uint64_t inv_n = goldilocks_inv_u64(N);

    auto* ctx = pil2::metal::get_context();
    pil2::metal::ntt_forward_metal(ctx, buf.data(), N, 1, roots.data(), N);
    pil2::metal::ntt_inverse_metal(ctx, buf.data(), N, 1, roots.data(), N, inv_n);

    for (uint64_t i = 0; i < N; ++i) {
        ASSERT_EQ(buf[i], orig[i])
            << "INTT(NTT(x)) != x at i=" << i
            << " orig=0x" << std::hex << orig[i]
            << " got=0x"  << buf[i];
    }
}

TEST(MetalINTT, BitExactMultiColN256_Ncols3) {
    // qDim=3 shape used by Starks::computeQ. N=256 keeps the test fast,
    // ncols=3 is the FIELD_EXTENSION dimension. In-place call pattern
    // matches the CPU prover.
    constexpr uint64_t LOG2_N = 8;
    constexpr uint64_t N      = 1ULL << LOG2_N;
    constexpr uint64_t NCOLS  = 3;

    std::mt19937_64 rng(0xC173E1FFULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> input_u64(N * NCOLS);
    for (auto& x : input_u64) x = dist(rng);

    std::vector<Goldilocks::Element> cpu_buf(N * NCOLS);
    for (uint64_t i = 0; i < N * NCOLS; ++i) cpu_buf[i] = Goldilocks::fromU64(input_u64[i]);

    NTT_Goldilocks ntt(N);
    ntt.INTT(cpu_buf.data(), cpu_buf.data(), N, NCOLS);

    std::vector<uint64_t> cpu_out(N * NCOLS);
    for (uint64_t i = 0; i < N * NCOLS; ++i) cpu_out[i] = Goldilocks::toU64(cpu_buf[i]);

    std::vector<uint64_t> gpu_buf = input_u64;
    std::vector<uint64_t> roots   = build_roots_table(LOG2_N);
    const uint64_t inv_n = goldilocks_inv_u64(N);

    pil2::metal::ntt_inverse_metal(pil2::metal::get_context(),
                                   gpu_buf.data(), N, NCOLS,
                                   roots.data(), N, inv_n);

    for (uint64_t i = 0; i < N * NCOLS; ++i) {
        ASSERT_EQ(gpu_buf[i], cpu_out[i])
            << "INTT(ncols=3) mismatch at i=" << i
            << " row=" << (i / NCOLS) << " col=" << (i % NCOLS)
            << " gpu=0x" << std::hex << gpu_buf[i]
            << " cpu=0x" << cpu_out[i];
    }
}

TEST(MetalINTT, BitExactVsScalarN1024) {
    // GPU INTT bit-exact vs NTT_Goldilocks::INTT. Validates that the GPU
    // reorder+scale path matches the scalar reference's element ordering
    // and 1/N convention precisely, not just up to round-trip identity.
    constexpr uint64_t LOG2_N = 10;
    constexpr uint64_t N      = 1ULL << LOG2_N;

    std::mt19937_64 rng(0xC5C1A11ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> input_u64(N);
    for (auto& x : input_u64) x = dist(rng);

    std::vector<Goldilocks::Element> cpu_buf(N);
    for (uint64_t i = 0; i < N; ++i) cpu_buf[i] = Goldilocks::fromU64(input_u64[i]);

    NTT_Goldilocks ntt(N);
    ntt.INTT(cpu_buf.data(), cpu_buf.data(), N);

    std::vector<uint64_t> cpu_out(N);
    for (uint64_t i = 0; i < N; ++i) cpu_out[i] = Goldilocks::toU64(cpu_buf[i]);

    std::vector<uint64_t> gpu_buf = input_u64;
    std::vector<uint64_t> roots   = build_roots_table(LOG2_N);
    const uint64_t inv_n = goldilocks_inv_u64(N);

    pil2::metal::ntt_inverse_metal(pil2::metal::get_context(),
                                   gpu_buf.data(), N, 1,
                                   roots.data(), N, inv_n);

    for (uint64_t i = 0; i < N; ++i) {
        ASSERT_EQ(gpu_buf[i], cpu_out[i])
            << "INTT mismatch at i=" << i
            << " gpu=0x" << std::hex << gpu_buf[i]
            << " cpu=0x" << cpu_out[i];
    }
}

TEST(MetalINTT, RoundTripN8) {
    // Smallest fused-INTT path: domain_pow=3, fused-only forward butterflies
    // + intt_reorder_scale tail.
    constexpr uint64_t LOG2_N = 3;
    constexpr uint64_t N      = 1ULL << LOG2_N;

    std::vector<uint64_t> orig(N);
    for (uint64_t i = 0; i < N; ++i) orig[i] = (i + 1) * 13ULL;

    std::vector<uint64_t> buf   = orig;
    std::vector<uint64_t> roots = build_roots_table(LOG2_N);
    const uint64_t inv_n = goldilocks_inv_u64(N);

    auto* ctx = pil2::metal::get_context();
    pil2::metal::ntt_forward_metal(ctx, buf.data(), N, 1, roots.data(), N);
    pil2::metal::ntt_inverse_metal(ctx, buf.data(), N, 1, roots.data(), N, inv_n);

    for (uint64_t i = 0; i < N; ++i) {
        ASSERT_EQ(buf[i], orig[i]) << "small-N round-trip mismatch at i=" << i;
    }
}

#else

TEST(MetalNTT, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

TEST(MetalINTT, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif // PIL2_HAS_METAL
