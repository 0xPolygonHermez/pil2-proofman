#include "../src/platform.hpp"

#include <gtest/gtest.h>

#if PIL2_HAS_METAL

#include "../src/metal/metal_context.hpp"
#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/poseidon2_goldilocks_constants.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr uint64_t GL_P = 0xFFFFFFFF00000001ULL;

// C8 (86 elements) + D8 (8 elements) converted to raw u64 for the Metal
// bridge.
std::vector<uint64_t> load_C8_u64() {
    std::vector<uint64_t> out(86);
    for (size_t i = 0; i < 86; ++i) {
        out[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::C8[i]);
    }
    return out;
}

std::vector<uint64_t> load_D8_u64() {
    std::vector<uint64_t> out(8);
    for (size_t i = 0; i < 8; ++i) {
        out[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::D8[i]);
    }
    return out;
}

std::vector<uint64_t> load_C12_u64() {
    std::vector<uint64_t> out(118);
    for (size_t i = 0; i < 118; ++i) out[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::C12[i]);
    return out;
}
std::vector<uint64_t> load_D12_u64() {
    std::vector<uint64_t> out(12);
    for (size_t i = 0; i < 12; ++i) out[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::D12[i]);
    return out;
}

std::vector<uint64_t> load_C16_u64() {
    std::vector<uint64_t> out(150);
    for (size_t i = 0; i < 150; ++i) out[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::C16[i]);
    return out;
}
std::vector<uint64_t> load_D16_u64() {
    std::vector<uint64_t> out(16);
    for (size_t i = 0; i < 16; ++i) out[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::D16[i]);
    return out;
}

} // namespace

TEST(MetalPoseidon2, PermuteW8SingleBlockBitExact) {
    // Input [0, 1, 2, ..., 7] — same vector used by the CPU gtests; useful
    // because any discrepancy is an obvious "CPU and GPU disagree about a
    // well-known fixture" signal.
    Goldilocks::Element in_scalar[8];
    Goldilocks::Element out_scalar[8];
    for (int i = 0; i < 8; ++i) in_scalar[i] = Goldilocks::fromU64(i);
    Poseidon2Goldilocks<8>::permute(out_scalar, in_scalar, Poseidon2Mode::Scalar);

    std::vector<uint64_t> in_u64(8), out_u64(8, 0);
    for (int i = 0; i < 8; ++i) in_u64[i] = Goldilocks::toU64(in_scalar[i]);

    auto C8 = load_C8_u64();
    auto D8 = load_D8_u64();

    pil2::metal::poseidon2_permute_w8_metal(pil2::metal::get_context(),
                                            out_u64.data(), in_u64.data(),
                                            /*count=*/1,
                                            C8.data(), D8.data());

    for (int i = 0; i < 8; ++i) {
        ASSERT_EQ(out_u64[i], Goldilocks::toU64(out_scalar[i]))
            << "single-block permute mismatch at i=" << i
            << " gpu=0x" << std::hex << out_u64[i]
            << " cpu=0x" << Goldilocks::toU64(out_scalar[i]);
    }
}

TEST(MetalPoseidon2, PermuteW8BatchBitExact) {
    // Batched version: run the GPU kernel over many independent random
    // states, scalar-permute each on the CPU, memcmp.
    constexpr uint64_t COUNT = 256;
    constexpr uint64_t W     = 8;

    std::mt19937_64 rng(0xC700D5E0ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> in_u64(COUNT * W);
    for (auto& x : in_u64) x = dist(rng);

    std::vector<uint64_t> cpu_out(COUNT * W);
    for (uint64_t i = 0; i < COUNT; ++i) {
        Goldilocks::Element in_el[W], out_el[W];
        for (uint64_t j = 0; j < W; ++j) in_el[j] = Goldilocks::fromU64(in_u64[i * W + j]);
        Poseidon2Goldilocks<8>::permute(out_el, in_el, Poseidon2Mode::Scalar);
        for (uint64_t j = 0; j < W; ++j) cpu_out[i * W + j] = Goldilocks::toU64(out_el[j]);
    }

    std::vector<uint64_t> gpu_out(COUNT * W, 0);
    auto C8 = load_C8_u64();
    auto D8 = load_D8_u64();

    pil2::metal::poseidon2_permute_w8_metal(pil2::metal::get_context(),
                                            gpu_out.data(), in_u64.data(),
                                            COUNT,
                                            C8.data(), D8.data());

    for (uint64_t i = 0; i < COUNT * W; ++i) {
        ASSERT_EQ(gpu_out[i], cpu_out[i])
            << "batch permute mismatch at flat-i=" << i
            << " (state=" << (i / W) << " col=" << (i % W) << ")";
    }
}

TEST(MetalPoseidon2, PermuteW12SingleAndBatchBitExact) {
    // Single block: input [0..11]
    {
        Goldilocks::Element in_el[12], out_el[12];
        for (int i = 0; i < 12; ++i) in_el[i] = Goldilocks::fromU64(i);
        Poseidon2Goldilocks<12>::permute(out_el, in_el, Poseidon2Mode::Scalar);

        std::vector<uint64_t> in_u64(12), out_u64(12, 0);
        for (int i = 0; i < 12; ++i) in_u64[i] = Goldilocks::toU64(in_el[i]);
        auto C12 = load_C12_u64();
        auto D12 = load_D12_u64();
        pil2::metal::poseidon2_permute_w12_metal(pil2::metal::get_context(),
                                                 out_u64.data(), in_u64.data(),
                                                 1, C12.data(), D12.data());
        for (int i = 0; i < 12; ++i) {
            ASSERT_EQ(out_u64[i], Goldilocks::toU64(out_el[i]))
                << "W=12 single-block mismatch at i=" << i;
        }
    }

    // Batch of 128 random states
    constexpr uint64_t COUNT = 128;
    std::mt19937_64 rng(0xC7120D0EULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> in_u64(COUNT * 12);
    for (auto& x : in_u64) x = dist(rng);

    std::vector<uint64_t> cpu_out(COUNT * 12);
    for (uint64_t i = 0; i < COUNT; ++i) {
        Goldilocks::Element in_el[12], out_el[12];
        for (uint64_t j = 0; j < 12; ++j) in_el[j] = Goldilocks::fromU64(in_u64[i * 12 + j]);
        Poseidon2Goldilocks<12>::permute(out_el, in_el, Poseidon2Mode::Scalar);
        for (uint64_t j = 0; j < 12; ++j) cpu_out[i * 12 + j] = Goldilocks::toU64(out_el[j]);
    }

    std::vector<uint64_t> gpu_out(COUNT * 12, 0);
    auto C12 = load_C12_u64();
    auto D12 = load_D12_u64();
    pil2::metal::poseidon2_permute_w12_metal(pil2::metal::get_context(),
                                             gpu_out.data(), in_u64.data(),
                                             COUNT, C12.data(), D12.data());

    for (uint64_t i = 0; i < COUNT * 12; ++i) {
        ASSERT_EQ(gpu_out[i], cpu_out[i])
            << "W=12 batch mismatch at flat-i=" << i;
    }
}

TEST(MetalPoseidon2, PermuteW16SingleAndBatchBitExact) {
    // Single block: input [0..15]
    {
        Goldilocks::Element in_el[16], out_el[16];
        for (int i = 0; i < 16; ++i) in_el[i] = Goldilocks::fromU64(i);
        Poseidon2Goldilocks<16>::permute(out_el, in_el, Poseidon2Mode::Scalar);

        std::vector<uint64_t> in_u64(16), out_u64(16, 0);
        for (int i = 0; i < 16; ++i) in_u64[i] = Goldilocks::toU64(in_el[i]);
        auto C16 = load_C16_u64();
        auto D16 = load_D16_u64();
        pil2::metal::poseidon2_permute_w16_metal(pil2::metal::get_context(),
                                                 out_u64.data(), in_u64.data(),
                                                 1, C16.data(), D16.data());
        for (int i = 0; i < 16; ++i) {
            ASSERT_EQ(out_u64[i], Goldilocks::toU64(out_el[i]))
                << "W=16 single-block mismatch at i=" << i;
        }
    }

    // Batch of 128 random states
    constexpr uint64_t COUNT = 128;
    std::mt19937_64 rng(0xC7160D0EULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> in_u64(COUNT * 16);
    for (auto& x : in_u64) x = dist(rng);

    std::vector<uint64_t> cpu_out(COUNT * 16);
    for (uint64_t i = 0; i < COUNT; ++i) {
        Goldilocks::Element in_el[16], out_el[16];
        for (uint64_t j = 0; j < 16; ++j) in_el[j] = Goldilocks::fromU64(in_u64[i * 16 + j]);
        Poseidon2Goldilocks<16>::permute(out_el, in_el, Poseidon2Mode::Scalar);
        for (uint64_t j = 0; j < 16; ++j) cpu_out[i * 16 + j] = Goldilocks::toU64(out_el[j]);
    }

    std::vector<uint64_t> gpu_out(COUNT * 16, 0);
    auto C16 = load_C16_u64();
    auto D16 = load_D16_u64();
    pil2::metal::poseidon2_permute_w16_metal(pil2::metal::get_context(),
                                             gpu_out.data(), in_u64.data(),
                                             COUNT, C16.data(), D16.data());

    for (uint64_t i = 0; i < COUNT * 16; ++i) {
        ASSERT_EQ(gpu_out[i], cpu_out[i])
            << "W=16 batch mismatch at flat-i=" << i;
    }
}

#else

TEST(MetalPoseidon2, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif // PIL2_HAS_METAL
