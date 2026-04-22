#include "../src/platform.hpp"

#include <gtest/gtest.h>

#if PIL2_HAS_METAL

#include "../src/metal/metal_context.hpp"
#include "../src/goldilocks_base_field.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr uint64_t GL_P = 0xFFFFFFFF00000001ULL;

} // namespace

// Sanity: allocate + free a Metal-backed shared buffer, read/write from the
// CPU side, confirm metal_is_shared_base tracks the pointer.
TEST(MetalSharedAlloc, AllocAccessAndFree) {
    const uint64_t bytes = 1024 * sizeof(uint64_t);
    void* p = pil2::metal::metal_alloc_shared(bytes);
    ASSERT_NE(p, nullptr);
    ASSERT_TRUE(pil2::metal::metal_is_shared_base(p));

    uint64_t* u = static_cast<uint64_t*>(p);
    for (uint64_t i = 0; i < 1024; ++i) u[i] = i * 0x9E3779B97F4A7C15ULL;
    for (uint64_t i = 0; i < 1024; ++i) {
        ASSERT_EQ(u[i], i * 0x9E3779B97F4A7C15ULL) << "CPU write/read at " << i;
    }

    pil2::metal::metal_free_shared(p);
    ASSERT_FALSE(pil2::metal::metal_is_shared_base(p));
}

// End-to-end: run a VM dispatch with aux_trace sourced from Metal-allocated
// shared memory (the zero-copy path), and compare against the same dispatch
// with aux_trace sourced from a plain std::vector (the scratch-memcpy
// fallback path). Must be bit-identical.
TEST(MetalSharedAlloc, VmBitExactViaRegisteredBufferVsFallback) {
    // Same bytecode shape as expr_vm's B.1 test — one row-varying trace
    // read + two numbers reads — but drive aux_trace/const_pols via the
    // shared allocator so the VM bridge's resolver hits the zero-copy
    // fast path. We compare against an identical run driven by malloc'd
    // vectors to confirm bit-for-bit equivalence.
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_AUX      = 2u;
    const uint32_t TYPE_TMP1     = bufferCommitsSize;
    const uint32_t TYPE_NUMBERS  = bufferCommitsSize + 3u;
    const uint32_t nThreads = 32;
    const uint32_t nColsAux = 3;

    std::vector<uint32_t> stage_offsets = {0u, 0u, 0u};
    std::vector<uint32_t> stage_ncols   = {0u, 0u, nColsAux};
    std::vector<int64_t>  next_strides  = {0};

    std::mt19937_64 rng(0xB1800001ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    // Scenario A: source aux_trace from ordinary std::vector (existing path).
    std::vector<uint64_t> aux_trace_vec(nThreads * nColsAux);
    for (auto& v : aux_trace_vec) v = dist(rng);
    std::vector<uint64_t> numbers(3);
    for (auto& v : numbers) v = dist(rng);

    std::vector<uint8_t>  ops  = {2, 2};
    std::vector<uint16_t> args = {
        // op 0: tmp3[0] = aux[r, 0..2] * numbers[0..2]  (cubic mul)
        2, 0,  (uint16_t)TYPE_AUX,      0, 0,  (uint16_t)TYPE_NUMBERS,   0, 0,
        // op 1 (last): dst = tmp3[0] + aux[r, 0..2]     (cubic add)
        0, 0,  (uint16_t)(bufferCommitsSize + 1u), 0, 0,  (uint16_t)TYPE_AUX, 0, 0,
    };

    std::vector<uint64_t> dst_fallback(nThreads * 3u, 0xDEADBEEFULL);
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(), numbers.data(),
                                 /*trace=*/nullptr, aux_trace_vec.data(),
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dst_fallback.data(),
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 (uint32_t)numbers.size(),
                                 /*trace_len_u64=*/0,
                                 (uint32_t)aux_trace_vec.size(),
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false,
                                 /*flat=*/{},
                                 /*dest_dim=*/3u);

    // Scenario B: allocate aux_trace through the Metal registry AND have
    // dst point into a Metal-allocated buffer — triggers the resolver on
    // both read (aux) and write (dst) paths. Content must match.
    const uint64_t aux_bytes = aux_trace_vec.size() * sizeof(uint64_t);
    const uint64_t dst_bytes = nThreads * 3u * sizeof(uint64_t);

    void* aux_shared = pil2::metal::metal_alloc_shared(aux_bytes);
    void* dst_shared = pil2::metal::metal_alloc_shared(dst_bytes);
    ASSERT_NE(aux_shared, nullptr);
    ASSERT_NE(dst_shared, nullptr);

    uint64_t* aux_u64 = static_cast<uint64_t*>(aux_shared);
    std::memcpy(aux_u64, aux_trace_vec.data(), aux_bytes);

    uint64_t* dst_u64 = static_cast<uint64_t*>(dst_shared);
    for (uint32_t i = 0; i < nThreads * 3u; ++i) dst_u64[i] = 0xDEADBEEFULL;

    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(), numbers.data(),
                                 /*trace=*/nullptr, aux_u64,
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dst_u64,
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 (uint32_t)numbers.size(),
                                 /*trace_len_u64=*/0,
                                 (uint32_t)aux_trace_vec.size(),
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false,
                                 /*flat=*/{},
                                 /*dest_dim=*/3u);

    for (uint32_t i = 0; i < nThreads * 3u; ++i) {
        ASSERT_EQ(dst_u64[i], dst_fallback[i])
            << "entry " << i
            << " shared=0x" << std::hex << dst_u64[i]
            << " fallback=0x" << dst_fallback[i];
    }

    pil2::metal::metal_free_shared(aux_shared);
    pil2::metal::metal_free_shared(dst_shared);
}

#else

TEST(MetalSharedAlloc, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif  // PIL2_HAS_METAL
