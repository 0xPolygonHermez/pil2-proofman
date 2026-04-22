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

// Byte-identical oracle: interprets the same 8-entry-per-op bytecode the
// Metal kernel runs, using the CPU Goldilocks primitives. Serves every
// expr_vm test variant — pass nullptr for buffers whose source type is
// not exercised by the bytecode.
// Mirrors pil2::metal::ExprVmFlatTables. Test-local so cpu_eval doesn't
// have to depend on the production struct layout.
struct TestFlat {
    const uint64_t* public_inputs   = nullptr;
    const uint64_t* air_values      = nullptr;
    const uint64_t* proof_values    = nullptr;
    const uint64_t* airgroup_values = nullptr;
    const uint64_t* challenges      = nullptr;
    const uint64_t* evals           = nullptr;
};

uint64_t cpu_eval(const std::vector<uint8_t>& ops,
                  const std::vector<uint16_t>& args,
                  const std::vector<uint64_t>& numbers,
                  const uint64_t* trace,
                  const uint64_t* aux_trace,
                  const uint64_t* const_pols,
                  const std::vector<uint32_t>& stage_offsets,
                  const std::vector<uint32_t>& stage_ncols,
                  const std::vector<int64_t>& next_strides,
                  uint32_t tid,
                  uint32_t domain_size,
                  uint32_t buffer_commits_size,
                  bool domain_extended,
                  const TestFlat& flat = {}) {
    uint64_t tmp1[64] = {};
    uint64_t last_result = 0;
    uint32_t i_args = 0;
    const uint64_t dom_mask = (uint64_t)(domain_size - 1u);
    for (uint32_t kk = 0; kk < ops.size(); ++kk) {
        uint32_t inner_op      = args[i_args + 0];
        uint32_t dest_slot     = args[i_args + 1];
        uint32_t typeA         = args[i_args + 2];
        uint32_t slotA         = args[i_args + 3];
        uint32_t rowOffsetIdxA = args[i_args + 4];
        uint32_t typeB         = args[i_args + 5];
        uint32_t slotB         = args[i_args + 6];
        uint32_t rowOffsetIdxB = args[i_args + 7];

        // Same wrap as the kernel: two's-complement add, mask by dom_size - 1.
        uint32_t rowA = (uint32_t)(((int64_t)tid + next_strides[rowOffsetIdxA]) & (int64_t)dom_mask);
        uint32_t rowB = (uint32_t)(((int64_t)tid + next_strides[rowOffsetIdxB]) & (int64_t)dom_mask);

        auto load = [&](uint32_t type, uint32_t slot, uint32_t row) -> uint64_t {
            if (type == 0u) {
                return const_pols[stage_offsets[0] + row * stage_ncols[0] + slot];
            }
            if (type == 1u && !domain_extended) {
                return trace[row * stage_ncols[1] + slot];
            }
            if (type == buffer_commits_size)       return tmp1[slot];
            if (type == buffer_commits_size + 2u)  return flat.public_inputs[slot];
            if (type == buffer_commits_size + 3u)  return numbers[slot];
            if (type == buffer_commits_size + 4u)  return flat.air_values[slot];
            if (type == buffer_commits_size + 5u)  return flat.proof_values[slot];
            if (type == buffer_commits_size + 6u)  return flat.airgroup_values[slot];
            if (type == buffer_commits_size + 7u)  return flat.challenges[slot];
            if (type == buffer_commits_size + 8u)  return flat.evals[slot];
            // Fallthrough: aux_trace. Covers type in [2, nStages+1] always,
            // plus type==1 when domain_extended.
            return aux_trace[stage_offsets[type] + row * stage_ncols[type] + slot];
        };
        uint64_t a = load(typeA, slotA, rowA);
        uint64_t b = load(typeB, slotB, rowB);

        Goldilocks::Element ae = Goldilocks::fromU64(a);
        Goldilocks::Element be = Goldilocks::fromU64(b);
        Goldilocks::Element re;
        switch (inner_op) {
            case 0: re = ae + be; break;
            case 1: re = ae - be; break;
            case 2: re = ae * be; break;
            default: re = be - ae; break;  // case 3
        }
        uint64_t r = Goldilocks::toU64(re);

        bool is_last = (kk + 1u == ops.size());
        if (is_last) last_result = r;
        else         tmp1[dest_slot] = r;

        i_args += 8;
    }
    return last_result;
}

} // namespace

// B.0 — tmp1 + numbers only. Row-invariant inputs; the whole point is to
// prove the host↔GPU wiring and the last-op-writes-dest convention.
TEST(MetalExprVmMin, ThreeOpTmp1AndNumbers) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_TMP1    = bufferCommitsSize;
    const uint32_t TYPE_NUMBERS = bufferCommitsSize + 3u;
    const uint32_t nThreads = 128;

    // Dummy stage tables — bytecode never dereferences them because no op
    // references type 1 or type in [2, nStages+1].
    std::vector<uint32_t> stage_offsets(1, 0u);
    std::vector<uint32_t> stage_ncols(1, 0u);
    std::vector<int64_t>  next_strides = {0};  // all ops use rowOffsetIdx=0

    for (uint64_t seed : {0x1111ULL, 0x2222ULL, 0x3333ULL}) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

        std::vector<uint64_t> numbers = { dist(rng), dist(rng), dist(rng), dist(rng) };

        std::vector<uint8_t>  ops  = {0, 0, 0};
        std::vector<uint16_t> args = {
            0, 0,  (uint16_t)TYPE_NUMBERS, 0, 0,  (uint16_t)TYPE_NUMBERS, 1, 0,
            2, 1,  (uint16_t)TYPE_TMP1,    0, 0,  (uint16_t)TYPE_NUMBERS, 2, 0,
            1, 0,  (uint16_t)TYPE_TMP1,    1, 0,  (uint16_t)TYPE_NUMBERS, 3, 0,
        };

        uint64_t expected = cpu_eval(ops, args, numbers,
                                     /*trace=*/nullptr, /*aux=*/nullptr,
                                     /*const_pols=*/nullptr,
                                     stage_offsets, stage_ncols,
                                     next_strides,
                                     /*tid=*/0, /*domain_size=*/nThreads,
                                     bufferCommitsSize,
                                     /*domain_extended=*/false);

        std::vector<uint64_t> dst(nThreads, 0xDEADBEEFULL);
        pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                     ops.data(), args.data(), numbers.data(),
                                     /*trace=*/nullptr, /*aux_trace=*/nullptr,
                                     /*const_pols=*/nullptr,
                                     stage_offsets.data(), stage_ncols.data(),
                                     next_strides.data(),
                                     dst.data(),
                                     (uint32_t)ops.size(), (uint32_t)args.size(),
                                     (uint32_t)numbers.size(),
                                     /*trace_len_u64=*/0, /*aux_trace_len_u64=*/0,
                                     /*const_pols_len_u64=*/0,
                                     (uint32_t)stage_offsets.size(),
                                     (uint32_t)next_strides.size(),
                                     nThreads, /*domain_size=*/nThreads,
                                     bufferCommitsSize,
                                     /*domain_extended=*/false);

        for (uint32_t t = 0; t < nThreads; ++t) {
            ASSERT_EQ(dst[t], expected)
                << "seed=0x" << std::hex << seed
                << " tid=" << std::dec << t;
        }
    }
}

// B.1 — trace + tmp1 + numbers. Row-varying trace proves per-thread tmp1
// independence.
TEST(MetalExprVmMin, FourOpWithTraceRowVarying) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_TRACE   = 1u;
    const uint32_t TYPE_TMP1    = bufferCommitsSize;
    const uint32_t TYPE_NUMBERS = bufferCommitsSize + 3u;
    const uint32_t nThreads = 64;
    const uint32_t nCols    = 2;

    // stage_ncols[1] = nCols=2 for the trace read; index 0 unused.
    std::vector<uint32_t> stage_offsets = {0u, 0u};
    std::vector<uint32_t> stage_ncols   = {0u, nCols};
    std::vector<int64_t>  next_strides  = {0};

    for (uint64_t seed : {0xB101ULL, 0xB102ULL, 0xB103ULL}) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

        std::vector<uint64_t> numbers = { dist(rng), dist(rng), dist(rng) };
        std::vector<uint64_t> trace(nThreads * nCols);
        for (auto& x : trace) x = dist(rng);

        std::vector<uint8_t>  ops  = {0, 0, 0, 0};
        std::vector<uint16_t> args = {
            0, 0,  (uint16_t)TYPE_TRACE,   0, 0,  (uint16_t)TYPE_NUMBERS, 0, 0,
            2, 1,  (uint16_t)TYPE_TMP1,    0, 0,  (uint16_t)TYPE_TRACE,   1, 0,
            1, 2,  (uint16_t)TYPE_TMP1,    1, 0,  (uint16_t)TYPE_NUMBERS, 1, 0,
            3, 0,  (uint16_t)TYPE_TMP1,    2, 0,  (uint16_t)TYPE_NUMBERS, 2, 0,
        };

        std::vector<uint64_t> dst(nThreads, 0xDEADBEEFULL);
        pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                     ops.data(), args.data(), numbers.data(),
                                     trace.data(), /*aux_trace=*/nullptr,
                                     /*const_pols=*/nullptr,
                                     stage_offsets.data(), stage_ncols.data(),
                                     next_strides.data(),
                                     dst.data(),
                                     (uint32_t)ops.size(), (uint32_t)args.size(),
                                     (uint32_t)numbers.size(),
                                     (uint32_t)trace.size(), /*aux_trace_len_u64=*/0,
                                     /*const_pols_len_u64=*/0,
                                     (uint32_t)stage_offsets.size(),
                                     (uint32_t)next_strides.size(),
                                     nThreads, /*domain_size=*/nThreads,
                                     bufferCommitsSize,
                                     /*domain_extended=*/false);

        for (uint32_t t = 0; t < nThreads; ++t) {
            uint64_t expected = cpu_eval(ops, args, numbers,
                                         trace.data(), /*aux=*/nullptr,
                                         /*const_pols=*/nullptr,
                                         stage_offsets, stage_ncols,
                                         next_strides,
                                         t, /*domain_size=*/nThreads,
                                         bufferCommitsSize,
                                         /*domain_extended=*/false);
            ASSERT_EQ(dst[t], expected)
                << "seed=0x" << std::hex << seed
                << " tid=" << std::dec << t;
        }
    }
}

// B.2 — trace + aux_trace + tmp1 + numbers in one program.
//
// Two stages:
//   stage 1 (trace)     : nCols=2, stage_offsets[1]=0
//   stage 2 (aux_trace) : nCols=3, stage_offsets[2]=0
//
// aux_trace is laid out row-major over the n_threads rows; stage 2 is the
// only stage in this test, so its base offset inside aux_trace is 0.
//
// Program (3 ops):
//   tmp[0] = trace[row,0] + aux[row,1]   (type 1 + type 2, add)
//   tmp[1] = tmp[0]       * numbers[0]   (tmp1 + numbers, mul)
//   dst    = aux[row,2]   - tmp[1]       (type 2 + tmp1, sub; last → dst)
//
// Correctness risks this targets:
//  - stage_offsets[] / stage_ncols[] indexed by type (not by stage number).
//  - aux_trace row stride distinct from trace row stride.
//  - aux_trace reads interleaved with trace reads in the same program.
TEST(MetalExprVmMin, ThreeOpTraceAndAuxTrace) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_TRACE   = 1u;
    const uint32_t TYPE_AUX_2   = 2u;
    const uint32_t TYPE_TMP1    = bufferCommitsSize;
    const uint32_t TYPE_NUMBERS = bufferCommitsSize + 3u;
    const uint32_t nThreads = 64;
    const uint32_t nColsTrace = 2;
    const uint32_t nColsAux   = 3;

    std::vector<uint32_t> stage_offsets = {0u, 0u, 0u};
    std::vector<uint32_t> stage_ncols   = {0u, nColsTrace, nColsAux};
    std::vector<int64_t>  next_strides  = {0};

    for (uint64_t seed : {0xB201ULL, 0xB202ULL, 0xB203ULL}) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

        std::vector<uint64_t> numbers = { dist(rng) };
        std::vector<uint64_t> trace(nThreads * nColsTrace);
        std::vector<uint64_t> aux_trace(nThreads * nColsAux);
        for (auto& x : trace)     x = dist(rng);
        for (auto& x : aux_trace) x = dist(rng);

        std::vector<uint8_t>  ops  = {0, 0, 0};
        std::vector<uint16_t> args = {
            // op 0: tmp[0] = trace[row, 0] + aux[row, 1]
            0, 0,  (uint16_t)TYPE_TRACE, 0, 0,  (uint16_t)TYPE_AUX_2,   1, 0,
            // op 1: tmp[1] = tmp[0] * numbers[0]
            2, 1,  (uint16_t)TYPE_TMP1,  0, 0,  (uint16_t)TYPE_NUMBERS, 0, 0,
            // op 2: dst = aux[row, 2] - tmp[1]
            1, 0,  (uint16_t)TYPE_AUX_2, 2, 0,  (uint16_t)TYPE_TMP1,    1, 0,
        };

        std::vector<uint64_t> dst(nThreads, 0xDEADBEEFULL);
        pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                     ops.data(), args.data(), numbers.data(),
                                     trace.data(), aux_trace.data(),
                                     /*const_pols=*/nullptr,
                                     stage_offsets.data(), stage_ncols.data(),
                                     next_strides.data(),
                                     dst.data(),
                                     (uint32_t)ops.size(), (uint32_t)args.size(),
                                     (uint32_t)numbers.size(),
                                     (uint32_t)trace.size(), (uint32_t)aux_trace.size(),
                                     /*const_pols_len_u64=*/0,
                                     (uint32_t)stage_offsets.size(),
                                     (uint32_t)next_strides.size(),
                                     nThreads, /*domain_size=*/nThreads,
                                     bufferCommitsSize,
                                     /*domain_extended=*/false);

        for (uint32_t t = 0; t < nThreads; ++t) {
            uint64_t expected = cpu_eval(ops, args, numbers,
                                         trace.data(), aux_trace.data(),
                                         /*const_pols=*/nullptr,
                                         stage_offsets, stage_ncols,
                                         next_strides,
                                         t, /*domain_size=*/nThreads,
                                         bufferCommitsSize,
                                         /*domain_extended=*/false);
            ASSERT_EQ(dst[t], expected)
                << "seed=0x" << std::hex << seed
                << " tid=" << std::dec << t
                << " got=0x" << std::hex << dst[t]
                << " want=0x" << expected;
        }
    }
}

// B.3 — domain_extended routes type==1 to aux_trace (stage_offsets[1] +
// row * stage_ncols[1] + slot) instead of the raw trace buffer.
//
// Same bytecode runs twice; trace and aux_trace hold DIFFERENT data at
// stage 1's logical slot. When domain_extended=false the answer must match
// the trace values; when =true it must match the aux_trace values (with a
// deliberately non-zero stage_offsets[1] so the offset math is exercised,
// not just the branch).
TEST(MetalExprVmMin, Type1RoutesByDomainExtended) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_TRACE   = 1u;
    const uint32_t TYPE_TMP1    = bufferCommitsSize;
    const uint32_t TYPE_NUMBERS = bufferCommitsSize + 3u;
    const uint32_t nThreads = 32;
    const uint32_t nColsStage1 = 2;
    const uint32_t stage1AuxOffset = 11;  // arbitrary non-zero, fits in 1 padding row + slack

    std::vector<uint32_t> stage_offsets = {0u, stage1AuxOffset};
    std::vector<uint32_t> stage_ncols   = {0u, nColsStage1};
    std::vector<int64_t>  next_strides  = {0};

    std::mt19937_64 rng(0xB301ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> numbers = { dist(rng) };

    // Raw trace buffer: nThreads rows × nColsStage1 cols.
    std::vector<uint64_t> trace(nThreads * nColsStage1);
    for (auto& x : trace) x = dist(rng);

    // aux_trace: stage1AuxOffset bytes of pad, then stage-1 data. The pad
    // region is deliberately filled with garbage distinct from trace to
    // catch a missing-offset bug (would read from aux_trace index 0).
    std::vector<uint64_t> aux_trace(stage1AuxOffset + nThreads * nColsStage1);
    for (auto& x : aux_trace) x = dist(rng);

    // Program: 2 ops, trivially different per row due to type-1 reads.
    //   tmp[0] = type1[row, 0] + numbers[0]
    //   dst    = type1[row, 1] - tmp[0]
    std::vector<uint8_t>  ops  = {0, 0};
    std::vector<uint16_t> args = {
        0, 0,  (uint16_t)TYPE_TRACE,   0, 0,  (uint16_t)TYPE_NUMBERS, 0, 0,
        1, 0,  (uint16_t)TYPE_TRACE,   1, 0,  (uint16_t)TYPE_TMP1,    0, 0,
    };

    for (bool domain_extended : {false, true}) {
        std::vector<uint64_t> dst(nThreads, 0xDEADBEEFULL);
        pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                     ops.data(), args.data(), numbers.data(),
                                     trace.data(), aux_trace.data(),
                                     /*const_pols=*/nullptr,
                                     stage_offsets.data(), stage_ncols.data(),
                                     next_strides.data(),
                                     dst.data(),
                                     (uint32_t)ops.size(), (uint32_t)args.size(),
                                     (uint32_t)numbers.size(),
                                     (uint32_t)trace.size(),
                                     (uint32_t)aux_trace.size(),
                                     /*const_pols_len_u64=*/0,
                                     (uint32_t)stage_offsets.size(),
                                     (uint32_t)next_strides.size(),
                                     nThreads, /*domain_size=*/nThreads,
                                     bufferCommitsSize,
                                     domain_extended);

        for (uint32_t t = 0; t < nThreads; ++t) {
            uint64_t expected = cpu_eval(ops, args, numbers,
                                         trace.data(), aux_trace.data(),
                                         /*const_pols=*/nullptr,
                                         stage_offsets, stage_ncols,
                                         next_strides,
                                         t, /*domain_size=*/nThreads,
                                         bufferCommitsSize,
                                         domain_extended);
            ASSERT_EQ(dst[t], expected)
                << "domain_extended=" << domain_extended
                << " tid=" << t
                << " got=0x" << std::hex << dst[t]
                << " want=0x" << expected;
        }
    }
}

// B.4 — const_pols (type == 0). Mixed with type==1 trace to catch a
// stride / offset mix-up (const_pols has its own stage_ncols[0] distinct
// from stage_ncols[1]).
//
// Program (3 ops):
//   tmp[0] = const_pols[row, 0] + trace[row, 0]     (type 0 + type 1, add)
//   tmp[1] = tmp[0]              * numbers[0]       (tmp1 + numbers, mul)
//   dst    = const_pols[row, 1] - tmp[1]            (type 0 + tmp1, sub)
TEST(MetalExprVmMin, ThreeOpConstPolsAndTrace) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_CONST   = 0u;
    const uint32_t TYPE_TRACE   = 1u;
    const uint32_t TYPE_TMP1    = bufferCommitsSize;
    const uint32_t TYPE_NUMBERS = bufferCommitsSize + 3u;
    const uint32_t nThreads = 64;
    const uint32_t nColsConst = 3;  // const_pols has 3 columns
    const uint32_t nColsTrace = 2;

    // Index 0 feeds const_pols, index 1 feeds trace.
    std::vector<uint32_t> stage_offsets = {0u, 0u};
    std::vector<uint32_t> stage_ncols   = {nColsConst, nColsTrace};
    std::vector<int64_t>  next_strides  = {0};

    for (uint64_t seed : {0xB401ULL, 0xB402ULL, 0xB403ULL}) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

        std::vector<uint64_t> numbers = { dist(rng) };
        std::vector<uint64_t> const_pols(nThreads * nColsConst);
        std::vector<uint64_t> trace(nThreads * nColsTrace);
        for (auto& x : const_pols) x = dist(rng);
        for (auto& x : trace)      x = dist(rng);

        std::vector<uint8_t>  ops  = {0, 0, 0};
        std::vector<uint16_t> args = {
            // op 0: tmp[0] = const_pols[row, 0] + trace[row, 0]
            0, 0,  (uint16_t)TYPE_CONST,  0, 0,  (uint16_t)TYPE_TRACE,   0, 0,
            // op 1: tmp[1] = tmp[0] * numbers[0]
            2, 1,  (uint16_t)TYPE_TMP1,   0, 0,  (uint16_t)TYPE_NUMBERS, 0, 0,
            // op 2: dst = const_pols[row, 1] - tmp[1]
            1, 0,  (uint16_t)TYPE_CONST,  1, 0,  (uint16_t)TYPE_TMP1,    1, 0,
        };

        std::vector<uint64_t> dst(nThreads, 0xDEADBEEFULL);
        pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                     ops.data(), args.data(), numbers.data(),
                                     trace.data(), /*aux_trace=*/nullptr,
                                     const_pols.data(),
                                     stage_offsets.data(), stage_ncols.data(),
                                     next_strides.data(),
                                     dst.data(),
                                     (uint32_t)ops.size(), (uint32_t)args.size(),
                                     (uint32_t)numbers.size(),
                                     (uint32_t)trace.size(),
                                     /*aux_trace_len_u64=*/0,
                                     (uint32_t)const_pols.size(),
                                     (uint32_t)stage_offsets.size(),
                                     (uint32_t)next_strides.size(),
                                     nThreads, /*domain_size=*/nThreads,
                                     bufferCommitsSize,
                                     /*domain_extended=*/false);

        for (uint32_t t = 0; t < nThreads; ++t) {
            uint64_t expected = cpu_eval(ops, args, numbers,
                                         trace.data(), /*aux=*/nullptr,
                                         const_pols.data(),
                                         stage_offsets, stage_ncols,
                                         next_strides,
                                         t, /*domain_size=*/nThreads,
                                         bufferCommitsSize,
                                         /*domain_extended=*/false);
            ASSERT_EQ(dst[t], expected)
                << "seed=0x" << std::hex << seed
                << " tid=" << std::dec << t
                << " got=0x" << std::hex << dst[t]
                << " want=0x" << expected;
        }
    }
}

// B.5 — all 6 flat-constant tables exercised in one program, mixed with
// a row-varying trace read so per-thread divergence is also verified.
// Each flat source sits in its own Metal buffer; a kernel bug that
// crossed two types (e.g. read challenges at the air_values buffer
// index) would diverge here for every row.
TEST(MetalExprVmMin, FlatTablesAllSixSources) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_TRACE    = 1u;
    const uint32_t TYPE_TMP1     = bufferCommitsSize;
    const uint32_t TYPE_PUBLIC   = bufferCommitsSize + 2u;
    const uint32_t TYPE_AIR      = bufferCommitsSize + 4u;
    const uint32_t TYPE_PROOF    = bufferCommitsSize + 5u;
    const uint32_t TYPE_GROUP    = bufferCommitsSize + 6u;
    const uint32_t TYPE_CHAL     = bufferCommitsSize + 7u;
    const uint32_t TYPE_EVALS    = bufferCommitsSize + 8u;
    const uint32_t nThreads = 32;
    const uint32_t nColsTrace = 2;

    std::vector<uint32_t> stage_offsets = {0u, 0u};
    std::vector<uint32_t> stage_ncols   = {0u, nColsTrace};
    std::vector<int64_t>  next_strides  = {0};

    std::mt19937_64 rng(0xB501ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> numbers(4);       for (auto& x : numbers)  x = dist(rng);
    std::vector<uint64_t> trace(nThreads * nColsTrace);
    for (auto& x : trace) x = dist(rng);

    std::vector<uint64_t> publics(4);       for (auto& x : publics)  x = dist(rng);
    std::vector<uint64_t> airvals(4);       for (auto& x : airvals)  x = dist(rng);
    std::vector<uint64_t> proofvals(4);     for (auto& x : proofvals) x = dist(rng);
    std::vector<uint64_t> groupvals(4);     for (auto& x : groupvals) x = dist(rng);
    std::vector<uint64_t> challenges(4);    for (auto& x : challenges) x = dist(rng);
    std::vector<uint64_t> evals(4);         for (auto& x : evals)    x = dist(rng);

    // 7-op program touching all 6 flat sources + trace + tmp1.
    //   tmp[0] = publics[0]  + trace[row, 0]
    //   tmp[1] = tmp[0]      * air_values[0]
    //   tmp[2] = tmp[1]      - proof_values[0]
    //   tmp[3] = tmp[2]      + airgroup_values[0]
    //   tmp[4] = tmp[3]      * challenges[0]
    //   tmp[5] = tmp[4]      - evals[0]
    //   dst    = evals[1]    + tmp[5]
    std::vector<uint8_t>  ops  = {0, 0, 0, 0, 0, 0, 0};
    std::vector<uint16_t> args = {
        0, 0,  (uint16_t)TYPE_PUBLIC, 0, 0,  (uint16_t)TYPE_TRACE, 0, 0,
        2, 1,  (uint16_t)TYPE_TMP1,   0, 0,  (uint16_t)TYPE_AIR,   0, 0,
        1, 2,  (uint16_t)TYPE_TMP1,   1, 0,  (uint16_t)TYPE_PROOF, 0, 0,
        0, 3,  (uint16_t)TYPE_TMP1,   2, 0,  (uint16_t)TYPE_GROUP, 0, 0,
        2, 4,  (uint16_t)TYPE_TMP1,   3, 0,  (uint16_t)TYPE_CHAL,  0, 0,
        1, 5,  (uint16_t)TYPE_TMP1,   4, 0,  (uint16_t)TYPE_EVALS, 0, 0,
        0, 0,  (uint16_t)TYPE_EVALS,  1, 0,  (uint16_t)TYPE_TMP1,  5, 0,
    };

    pil2::metal::ExprVmFlatTables flat;
    flat.public_inputs           = publics.data();
    flat.air_values              = airvals.data();
    flat.proof_values            = proofvals.data();
    flat.airgroup_values         = groupvals.data();
    flat.challenges              = challenges.data();
    flat.evals                   = evals.data();
    flat.public_inputs_len_u64   = (uint32_t)publics.size();
    flat.air_values_len_u64      = (uint32_t)airvals.size();
    flat.proof_values_len_u64    = (uint32_t)proofvals.size();
    flat.airgroup_values_len_u64 = (uint32_t)groupvals.size();
    flat.challenges_len_u64      = (uint32_t)challenges.size();
    flat.evals_len_u64           = (uint32_t)evals.size();

    std::vector<uint64_t> dst(nThreads, 0xDEADBEEFULL);
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(), numbers.data(),
                                 trace.data(), /*aux_trace=*/nullptr,
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dst.data(),
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 (uint32_t)numbers.size(),
                                 (uint32_t)trace.size(),
                                 /*aux_trace_len_u64=*/0,
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false,
                                 flat);

    TestFlat test_flat;
    test_flat.public_inputs   = publics.data();
    test_flat.air_values      = airvals.data();
    test_flat.proof_values    = proofvals.data();
    test_flat.airgroup_values = groupvals.data();
    test_flat.challenges      = challenges.data();
    test_flat.evals           = evals.data();

    for (uint32_t t = 0; t < nThreads; ++t) {
        uint64_t expected = cpu_eval(ops, args, numbers,
                                     trace.data(), /*aux=*/nullptr,
                                     /*const_pols=*/nullptr,
                                     stage_offsets, stage_ncols,
                                     next_strides,
                                     t, /*domain_size=*/nThreads,
                                     bufferCommitsSize,
                                     /*domain_extended=*/false,
                                     test_flat);
        ASSERT_EQ(dst[t], expected)
            << "tid=" << t
            << " got=0x" << std::hex << dst[t]
            << " want=0x" << expected;
    }
}

// B.6 — next-row and previous-row reads with cyclic wrap. 4-row domain
// with distinct trace values per row; the program computes
//   dst[row] = trace[row, 0] + trace[row+1, 0] - trace[row-1, 0]
// using next_strides = {0, 1, -1} (idx 0 = current, 1 = +1, 2 = -1).
// Every boundary case is hit: tid=0 reading row-1 must wrap to row 3,
// tid=3 reading row+1 must wrap to row 0. A missing wrap, wrong mask,
// or crossed idx-lookup would diverge on at least one row.
TEST(MetalExprVmMin, NextPrevRowWrap) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_TRACE = 1u;
    const uint32_t TYPE_TMP1  = bufferCommitsSize;
    const uint32_t nThreads = 4;          // also == domain_size (pow of 2)
    const uint32_t nColsTrace = 1;

    std::vector<uint32_t> stage_offsets = {0u, 0u};
    std::vector<uint32_t> stage_ncols   = {0u, nColsTrace};
    std::vector<int64_t>  next_strides  = {0, 1, -1};  // idx 0/1/2

    std::mt19937_64 rng(0xB601ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    // Deliberately distinct values per row so wrap errors are visible.
    std::vector<uint64_t> trace(nThreads * nColsTrace);
    for (auto& x : trace) x = dist(rng);
    std::vector<uint64_t> numbers = { dist(rng) };

    // Two ops:
    //   tmp[0] = trace[row, 0] + trace[row+1, 0]
    //   dst    = tmp[0]        - trace[row-1, 0]
    std::vector<uint8_t>  ops  = {0, 0};
    std::vector<uint16_t> args = {
        // op 0: rowOffsetIdx 0 (A, current row) and 1 (B, +1)
        0, 0,  (uint16_t)TYPE_TRACE, 0, 0,  (uint16_t)TYPE_TRACE, 0, 1,
        // op 1: rowOffsetIdx 0 (A, tmp1 — ignored anyway) and 2 (B, -1)
        1, 0,  (uint16_t)TYPE_TMP1,  0, 0,  (uint16_t)TYPE_TRACE, 0, 2,
    };

    std::vector<uint64_t> dst(nThreads, 0xDEADBEEFULL);
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(), numbers.data(),
                                 trace.data(), /*aux_trace=*/nullptr,
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dst.data(),
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 (uint32_t)numbers.size(),
                                 (uint32_t)trace.size(),
                                 /*aux_trace_len_u64=*/0,
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false);

    for (uint32_t t = 0; t < nThreads; ++t) {
        uint64_t expected = cpu_eval(ops, args, numbers,
                                     trace.data(), /*aux=*/nullptr,
                                     /*const_pols=*/nullptr,
                                     stage_offsets, stage_ncols,
                                     next_strides,
                                     t, /*domain_size=*/nThreads,
                                     bufferCommitsSize,
                                     /*domain_extended=*/false);
        ASSERT_EQ(dst[t], expected)
            << "tid=" << t
            << " got=0x" << std::hex << dst[t]
            << " want=0x" << expected;
    }
}

// B.7 — outer op 2 (dim3 × dim3 → dim3). First cubic path in the VM.
//
// 2-op program:
//   tmp3[0] = aux_trace[row, 0..2] * numbers[0..2]   (outer 2 + inner 2)
//   dst     = tmp3[0]              + aux_trace[row, 3..5]   (outer 2 + inner 0)
//
// This exercises: cubic read from aux_trace at two distinct cubic
// columns (slots 0 and 3 within the same row), cubic read from a flat
// source (numbers, 3 consecutive slots), cubic write to tmp3, cubic
// read from tmp3 back into operand A, cubic dst write of 3 u64s per
// thread. A kernel bug in any of those paths diverges from the
// per-row Goldilocks3 oracle on at least one thread.
TEST(MetalExprVmMin, OuterOp2CubicMulAdd) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_AUX      = 2u;
    const uint32_t TYPE_TMP3     = bufferCommitsSize + 1u;
    const uint32_t TYPE_NUMBERS  = bufferCommitsSize + 3u;
    const uint32_t nThreads = 16;
    const uint32_t nColsAux = 6;   // 2 cubic columns per row

    std::vector<uint32_t> stage_offsets = {0u, 0u, 0u};
    std::vector<uint32_t> stage_ncols   = {0u, 0u, nColsAux};
    std::vector<int64_t>  next_strides  = {0};

    std::mt19937_64 rng(0xB701ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> aux_trace(nThreads * nColsAux);
    for (auto& x : aux_trace) x = dist(rng);
    std::vector<uint64_t> numbers(3);
    for (auto& x : numbers) x = dist(rng);

    std::vector<uint8_t>  ops  = {2, 2};
    std::vector<uint16_t> args = {
        // op 0: outer=2, inner=2 (mul): tmp3[0] = aux[r, 0..2] * numbers[0..2]
        2, 0,  (uint16_t)TYPE_AUX, 0, 0,  (uint16_t)TYPE_NUMBERS, 0, 0,
        // op 1: outer=2, inner=0 (add), last: dst = tmp3[0] + aux[r, 3..5]
        0, 0,  (uint16_t)TYPE_TMP3, 0, 0,  (uint16_t)TYPE_AUX,    3, 0,
    };

    std::vector<uint64_t> dst(nThreads * 3u, 0xDEADBEEFULL);
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(), numbers.data(),
                                 /*trace=*/nullptr, aux_trace.data(),
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dst.data(),
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 (uint32_t)numbers.size(),
                                 /*trace_len_u64=*/0,
                                 (uint32_t)aux_trace.size(),
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false,
                                 /*flat=*/{},
                                 /*dest_dim=*/3u);

    for (uint32_t t = 0; t < nThreads; ++t) {
        Goldilocks3::Element aux_c0, num_c;
        for (int c = 0; c < 3; ++c) {
            aux_c0[c] = Goldilocks::fromU64(aux_trace[t * nColsAux + c]);
            num_c[c]  = Goldilocks::fromU64(numbers[c]);
        }
        Goldilocks3::Element tmp0;
        Goldilocks3::mul(tmp0, aux_c0, num_c);

        Goldilocks3::Element aux_c1;
        for (int c = 0; c < 3; ++c) {
            aux_c1[c] = Goldilocks::fromU64(aux_trace[t * nColsAux + 3 + c]);
        }
        Goldilocks3::Element res;
        Goldilocks3::add(res, tmp0, aux_c1);

        for (int c = 0; c < 3; ++c) {
            uint64_t expected = Goldilocks::toU64(res[c]);
            ASSERT_EQ(dst[t * 3u + c], expected)
                << "tid=" << t << " c=" << c
                << " got=0x" << std::hex << dst[t * 3u + c]
                << " want=0x" << expected;
        }
    }
}

// B.8 — outer op 1 (dim3 × dim1 → dim3). Cubic operand A, scalar B.
//
// 2-op program mixing cubic and scalar sources:
//   tmp3[0] = aux[row, 0..2] + numbers[0]      (outer 1, inner 0: op_31 add)
//   dst     = tmp3[0]        * challenges[0]   (outer 1, inner 2: op_31 mul)
//
// op_31 semantics (mirrors Goldilocks3::op_31_pack):
//   add/sub: only c0 touched; c1, c2 pass through from A
//   mul    : every component multiplied by scalar
// A kernel bug in component routing (e.g. widening scalar to [s, 0, 0] and
// calling full gl3_op instead of gl3_op_31) would change c1/c2 for the
// add/sub path and diverge immediately.
TEST(MetalExprVmMin, OuterOp1CubicScalarMulAdd) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_AUX        = 2u;
    const uint32_t TYPE_TMP3       = bufferCommitsSize + 1u;
    const uint32_t TYPE_NUMBERS    = bufferCommitsSize + 3u;
    const uint32_t TYPE_CHALLENGES = bufferCommitsSize + 7u;
    const uint32_t nThreads = 16;
    const uint32_t nColsAux = 3;  // one cubic column

    std::vector<uint32_t> stage_offsets = {0u, 0u, 0u};
    std::vector<uint32_t> stage_ncols   = {0u, 0u, nColsAux};
    std::vector<int64_t>  next_strides  = {0};

    std::mt19937_64 rng(0xB801ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> aux_trace(nThreads * nColsAux);
    for (auto& x : aux_trace) x = dist(rng);
    std::vector<uint64_t> numbers    = { dist(rng) };
    std::vector<uint64_t> challenges = { dist(rng) };

    std::vector<uint8_t>  ops  = {1, 1};
    std::vector<uint16_t> args = {
        // op 0: outer=1 inner=0 (add): tmp3[0] = aux[r, 0..2] + numbers[0]
        0, 0,  (uint16_t)TYPE_AUX,  0, 0,  (uint16_t)TYPE_NUMBERS,    0, 0,
        // op 1: outer=1 inner=2 (mul), last: dst = tmp3[0] * challenges[0]
        2, 0,  (uint16_t)TYPE_TMP3, 0, 0,  (uint16_t)TYPE_CHALLENGES, 0, 0,
    };

    pil2::metal::ExprVmFlatTables flat;
    flat.challenges            = challenges.data();
    flat.challenges_len_u64    = (uint32_t)challenges.size();

    std::vector<uint64_t> dst(nThreads * 3u, 0xDEADBEEFULL);
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(), numbers.data(),
                                 /*trace=*/nullptr, aux_trace.data(),
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dst.data(),
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 (uint32_t)numbers.size(),
                                 /*trace_len_u64=*/0,
                                 (uint32_t)aux_trace.size(),
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false,
                                 flat,
                                 /*dest_dim=*/3u);

    Goldilocks::Element n0 = Goldilocks::fromU64(numbers[0]);
    Goldilocks::Element ch0 = Goldilocks::fromU64(challenges[0]);

    for (uint32_t t = 0; t < nThreads; ++t) {
        Goldilocks3::Element aux_c;
        for (int c = 0; c < 3; ++c) {
            aux_c[c] = Goldilocks::fromU64(aux_trace[t * nColsAux + c]);
        }
        // op_31 add: only c0 gets the scalar
        Goldilocks3::Element tmp0 = { aux_c[0] + n0, aux_c[1], aux_c[2] };
        // op_31 mul: every component scaled
        Goldilocks3::Element res  = { tmp0[0] * ch0, tmp0[1] * ch0, tmp0[2] * ch0 };

        for (int c = 0; c < 3; ++c) {
            uint64_t expected = Goldilocks::toU64(res[c]);
            ASSERT_EQ(dst[t * 3u + c], expected)
                << "tid=" << t << " c=" << c
                << " got=0x" << std::hex << dst[t * 3u + c]
                << " want=0x" << expected;
        }
    }
}

// B.9 — customCommits (ROM-like per-AIR fixed columns). Two commits with
// distinct nCols and distinct offsets inside a shared data buffer. The
// bytecode mixes reads from both commits + tmp1 + numbers; any crossed
// offset/ncols index would diverge for at least one row.
//
// Setup (matches real AIR conventions):
//   nStages = 4, n_custom_commits = 2  →  bufferCommitsSize = 10
//   Custom commit 0 lives at type = 8, nCols = 2, data offset = 0
//   Custom commit 1 lives at type = 9, nCols = 3, data offset = nThreads * 2
//
// Program:
//   tmp[0] = custom0[row, 0] + custom1[row, 0]
//   tmp[1] = tmp[0]          * custom1[row, 2]    (different slot of commit 1)
//   dst    = tmp[1]          - custom0[row, 1]    (different slot of commit 0)
TEST(MetalExprVmMin, CustomCommitsTwoCommits) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_CUSTOM0 = bufferCommitsSize - 2u;  // = 8
    const uint32_t TYPE_CUSTOM1 = bufferCommitsSize - 1u;  // = 9
    const uint32_t TYPE_TMP1    = bufferCommitsSize;
    const uint32_t nThreads = 32;
    const uint32_t nColsC0 = 2;
    const uint32_t nColsC1 = 3;
    const uint32_t offC0 = 0;
    const uint32_t offC1 = nThreads * nColsC0;  // after commit 0's region

    // Stage tables minimal — bytecode doesn't reference trace / aux_trace.
    std::vector<uint32_t> stage_offsets = {0u};
    std::vector<uint32_t> stage_ncols   = {0u};
    std::vector<int64_t>  next_strides  = {0};

    std::vector<uint32_t> custom_offsets = {offC0, offC1};
    std::vector<uint32_t> custom_ncols   = {nColsC0, nColsC1};

    std::mt19937_64 rng(0xB901ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    // Concatenated data: [commit 0 rows | commit 1 rows].
    std::vector<uint64_t> custom_data(nThreads * nColsC0 + nThreads * nColsC1);
    for (auto& x : custom_data) x = dist(rng);

    std::vector<uint64_t> numbers;  // unused but bridge wants a non-null pointer? default is ok

    std::vector<uint8_t>  ops  = {0, 0, 0};
    std::vector<uint16_t> args = {
        // op 0: tmp[0] = custom0[row, 0] + custom1[row, 0]
        0, 0,  (uint16_t)TYPE_CUSTOM0, 0, 0,  (uint16_t)TYPE_CUSTOM1, 0, 0,
        // op 1: tmp[1] = tmp[0] * custom1[row, 2]
        2, 1,  (uint16_t)TYPE_TMP1,    0, 0,  (uint16_t)TYPE_CUSTOM1, 2, 0,
        // op 2: dst = tmp[1] - custom0[row, 1]
        1, 0,  (uint16_t)TYPE_TMP1,    1, 0,  (uint16_t)TYPE_CUSTOM0, 1, 0,
    };

    pil2::metal::ExprVmCustomCommits custom;
    custom.data         = custom_data.data();
    custom.offsets      = custom_offsets.data();
    custom.ncols        = custom_ncols.data();
    custom.data_len_u64 = (uint32_t)custom_data.size();
    custom.count        = 2u;

    std::vector<uint64_t> dst(nThreads, 0xDEADBEEFULL);
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(),
                                 /*numbers=*/nullptr,
                                 /*trace=*/nullptr, /*aux_trace=*/nullptr,
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dst.data(),
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 /*n_numbers=*/0,
                                 /*trace_len_u64=*/0,
                                 /*aux_trace_len_u64=*/0,
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false,
                                 /*flat=*/{},
                                 /*dest_dim=*/1u,
                                 custom);

    // Oracle — direct compute per row. tmp1 is per-thread, no cross-row.
    for (uint32_t t = 0; t < nThreads; ++t) {
        auto read_c0 = [&](uint32_t slot) {
            return Goldilocks::fromU64(custom_data[offC0 + t * nColsC0 + slot]);
        };
        auto read_c1 = [&](uint32_t slot) {
            return Goldilocks::fromU64(custom_data[offC1 + t * nColsC1 + slot]);
        };
        Goldilocks::Element tmp0 = read_c0(0) + read_c1(0);
        Goldilocks::Element tmp1 = tmp0       * read_c1(2);
        Goldilocks::Element res  = tmp1       - read_c0(1);
        uint64_t expected = Goldilocks::toU64(res);
        ASSERT_EQ(dst[t], expected)
            << "tid=" << t
            << " got=0x" << std::hex << dst[t]
            << " want=0x" << expected;
    }
}

// B.10 — proverHelpers (type == nStages+2). slot==0 reads x_current[row];
// slot>=1 reads zi[(slot-1) * domain_size + row]. Both paths exercised
// in one program, mixed with numbers. Per-thread distinct since both
// sources are row-varying.
//
// Setup: nStages=4, no custom commits → bufferCommitsSize=8, proverHelpers
// at type==6. With 2 boundaries, zi buffer has 2*domain_size u64s.
TEST(MetalExprVmMin, ProverHelpersXAndZi) {
    const uint32_t bufferCommitsSize = 8u;   // nStages+4, no custom commits
    const uint32_t TYPE_PROVER  = 6u;         // == nStages+2
    const uint32_t TYPE_TMP1    = bufferCommitsSize;
    const uint32_t TYPE_NUMBERS = bufferCommitsSize + 3u;
    const uint32_t nThreads = 32;
    const uint32_t nBoundaries = 2;

    std::vector<uint32_t> stage_offsets = {0u};
    std::vector<uint32_t> stage_ncols   = {0u};
    std::vector<int64_t>  next_strides  = {0};

    std::mt19937_64 rng(0xBA01ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> x_current(nThreads);
    std::vector<uint64_t> zi(nBoundaries * nThreads);
    for (auto& v : x_current) v = dist(rng);
    for (auto& v : zi)        v = dist(rng);
    std::vector<uint64_t> numbers = { dist(rng) };

    // 3-op program:
    //   tmp[0] = proverHelpers[slot=0, row]    + numbers[0]    → x_current + num
    //   tmp[1] = tmp[0]                         * proverHelpers[slot=1, row]  → * zi[0, row]
    //   dst    = tmp[1]                         - proverHelpers[slot=2, row]  → - zi[1, row]
    std::vector<uint8_t>  ops  = {0, 0, 0};
    std::vector<uint16_t> args = {
        0, 0,  (uint16_t)TYPE_PROVER, 0, 0,  (uint16_t)TYPE_NUMBERS, 0, 0,
        2, 1,  (uint16_t)TYPE_TMP1,   0, 0,  (uint16_t)TYPE_PROVER,  1, 0,
        1, 0,  (uint16_t)TYPE_TMP1,   1, 0,  (uint16_t)TYPE_PROVER,  2, 0,
    };

    pil2::metal::ExprVmProverHelpers ph;
    ph.x_current           = x_current.data();
    ph.zi                  = zi.data();
    ph.x_current_len_u64   = (uint32_t)x_current.size();
    ph.zi_len_u64          = (uint32_t)zi.size();

    std::vector<uint64_t> dst(nThreads, 0xDEADBEEFULL);
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(), numbers.data(),
                                 /*trace=*/nullptr, /*aux_trace=*/nullptr,
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dst.data(),
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 (uint32_t)numbers.size(),
                                 /*trace_len_u64=*/0,
                                 /*aux_trace_len_u64=*/0,
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false,
                                 /*flat=*/{},
                                 /*dest_dim=*/1u,
                                 /*custom=*/{},
                                 ph);

    Goldilocks::Element n0 = Goldilocks::fromU64(numbers[0]);
    for (uint32_t t = 0; t < nThreads; ++t) {
        Goldilocks::Element x  = Goldilocks::fromU64(x_current[t]);
        Goldilocks::Element z0 = Goldilocks::fromU64(zi[0 * nThreads + t]);
        Goldilocks::Element z1 = Goldilocks::fromU64(zi[1 * nThreads + t]);
        Goldilocks::Element tmp0 = x + n0;
        Goldilocks::Element tmp1 = tmp0 * z0;
        Goldilocks::Element res  = tmp1 - z1;
        uint64_t expected = Goldilocks::toU64(res);
        ASSERT_EQ(dst[t], expected)
            << "tid=" << t
            << " got=0x" << std::hex << dst[t]
            << " want=0x" << expected;
    }
}

// B.11 — xi source (type == nStages+3). Prover compute-on-read: kernel
// computes gl3_inv(x_current[row] - xis[slot]) per thread, treating x
// as a cubic widening (x, 0, 0).
//
// 2-op program exercising xi as a cubic operand:
//   tmp3[0] = xi[0] * numbers[0..2]          (outer 2, inner 2)
//   dst     = tmp3[0] + aux_trace[row, 0..2] (outer 2, inner 0)
//
// x_current is row-varying so every thread computes a distinct xdivxsub,
// catching any xi bug that still happens to be constant-per-row.
TEST(MetalExprVmMin, XiReadAndCubicMul) {
    const uint32_t nStages = 4u;
    const uint32_t bufferCommitsSize = nStages + 4u;  // no custom commits → 8
    const uint32_t TYPE_AUX      = 2u;
    const uint32_t TYPE_XI       = nStages + 3u;       // = 7
    const uint32_t TYPE_TMP3     = bufferCommitsSize + 1u;
    const uint32_t TYPE_NUMBERS  = bufferCommitsSize + 3u;
    const uint32_t nThreads = 16;
    const uint32_t nColsAux = 3;

    std::vector<uint32_t> stage_offsets = {0u, 0u, 0u};
    std::vector<uint32_t> stage_ncols   = {0u, 0u, nColsAux};
    std::vector<int64_t>  next_strides  = {0};

    std::mt19937_64 rng(0xB11B0001ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> x_current(nThreads);
    for (auto& v : x_current) v = dist(rng);

    // Single opening point for simplicity; xis is 3 u64.
    std::vector<uint64_t> xis = { dist(rng), dist(rng), dist(rng) };

    std::vector<uint64_t> aux_trace(nThreads * nColsAux);
    for (auto& x : aux_trace) x = dist(rng);

    std::vector<uint64_t> numbers(3);
    for (auto& x : numbers) x = dist(rng);

    std::vector<uint8_t>  ops  = {2, 2};
    std::vector<uint16_t> args = {
        // op 0: tmp3[0] = xi[slot=0] * numbers[0..2]   (outer 2, inner 2)
        2, 0,  (uint16_t)TYPE_XI,   0, 0,  (uint16_t)TYPE_NUMBERS, 0, 0,
        // op 1: dst = tmp3[0] + aux[row, 0..2]         (outer 2, inner 0)
        0, 0,  (uint16_t)TYPE_TMP3, 0, 0,  (uint16_t)TYPE_AUX,     0, 0,
    };

    pil2::metal::ExprVmProverHelpers ph;
    ph.x_current         = x_current.data();
    ph.xis               = xis.data();
    ph.x_current_len_u64 = (uint32_t)x_current.size();
    ph.xis_len_u64       = (uint32_t)xis.size();

    std::vector<uint64_t> dst(nThreads * 3u, 0xDEADBEEFULL);
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(), numbers.data(),
                                 /*trace=*/nullptr, aux_trace.data(),
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dst.data(),
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 (uint32_t)numbers.size(),
                                 /*trace_len_u64=*/0,
                                 (uint32_t)aux_trace.size(),
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false,
                                 /*flat=*/{},
                                 /*dest_dim=*/3u,
                                 /*custom=*/{},
                                 ph);

    // Oracle: per row, compute cubic (x_row - xis), cubic inverse, cubic
    // mul with numbers-widened, then cubic add with aux_trace cubic.
    Goldilocks3::Element xi_cubic;
    for (int c = 0; c < 3; ++c) xi_cubic[c] = Goldilocks::fromU64(xis[c]);
    Goldilocks3::Element num_cubic;
    for (int c = 0; c < 3; ++c) num_cubic[c] = Goldilocks::fromU64(numbers[c]);

    for (uint32_t t = 0; t < nThreads; ++t) {
        Goldilocks3::Element x_wide;
        x_wide[0] = Goldilocks::fromU64(x_current[t]);
        x_wide[1] = Goldilocks::zero();
        x_wide[2] = Goldilocks::zero();
        Goldilocks3::Element diff;
        Goldilocks3::sub(diff, x_wide, xi_cubic);
        Goldilocks3::Element xdivxsub;
        Goldilocks3::inv(xdivxsub, diff);
        Goldilocks3::Element tmp0;
        Goldilocks3::mul(tmp0, xdivxsub, num_cubic);

        Goldilocks3::Element aux_c;
        for (int c = 0; c < 3; ++c) aux_c[c] = Goldilocks::fromU64(aux_trace[t * nColsAux + c]);
        Goldilocks3::Element res;
        Goldilocks3::add(res, tmp0, aux_c);

        for (int c = 0; c < 3; ++c) {
            uint64_t expected = Goldilocks::toU64(res[c]);
            ASSERT_EQ(dst[t * 3u + c], expected)
                << "tid=" << t << " c=" << c
                << " got=0x" << std::hex << dst[t * 3u + c]
                << " want=0x" << expected;
        }
    }
}

// B.12 — dest_inverse post-op. The VM writes a raw cubic result per
// thread; the bridge chains a second dispatch that inverts each cubic
// row in place. Covers inverse-polynomial expressions (lookup /
// permutation witnesses) without a host round-trip between kernels.
//
// 1-op program: dst = aux[row, 0..2] + numbers[0..2]  (outer 2, inner 0).
// Then dest_inverse=true → kernel post-processes dst to be
// Goldilocks3::inv of that sum.
TEST(MetalExprVmMin, DestInverseCubic) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_AUX      = 2u;
    const uint32_t TYPE_NUMBERS  = bufferCommitsSize + 3u;
    const uint32_t nThreads = 16;
    const uint32_t nColsAux = 3;

    std::vector<uint32_t> stage_offsets = {0u, 0u, 0u};
    std::vector<uint32_t> stage_ncols   = {0u, 0u, nColsAux};
    std::vector<int64_t>  next_strides  = {0};

    std::mt19937_64 rng(0xB12C0001ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> aux_trace(nThreads * nColsAux);
    for (auto& x : aux_trace) x = dist(rng);
    std::vector<uint64_t> numbers(3);
    for (auto& x : numbers) x = dist(rng);

    std::vector<uint8_t>  ops  = {2};
    std::vector<uint16_t> args = {
        // op 0: dst = aux[r, 0..2] + numbers[0..2]  (outer 2, inner 0, last)
        0, 0,  (uint16_t)TYPE_AUX, 0, 0,  (uint16_t)TYPE_NUMBERS, 0, 0,
    };

    std::vector<uint64_t> dst(nThreads * 3u, 0xDEADBEEFULL);
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(), numbers.data(),
                                 /*trace=*/nullptr, aux_trace.data(),
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dst.data(),
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 (uint32_t)numbers.size(),
                                 /*trace_len_u64=*/0,
                                 (uint32_t)aux_trace.size(),
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false,
                                 /*flat=*/{},
                                 /*dest_dim=*/3u,
                                 /*custom=*/{},
                                 /*prover_helpers=*/{},
                                 /*dest_inverse=*/true);

    // Oracle: compute cubic sum then cubic inverse.
    Goldilocks3::Element num_c;
    for (int c = 0; c < 3; ++c) num_c[c] = Goldilocks::fromU64(numbers[c]);

    for (uint32_t t = 0; t < nThreads; ++t) {
        Goldilocks3::Element aux_c;
        for (int c = 0; c < 3; ++c) aux_c[c] = Goldilocks::fromU64(aux_trace[t * nColsAux + c]);
        Goldilocks3::Element sum;
        Goldilocks3::add(sum, aux_c, num_c);
        Goldilocks3::Element res;
        Goldilocks3::inv(res, sum);

        for (int c = 0; c < 3; ++c) {
            uint64_t expected = Goldilocks::toU64(res[c]);
            ASSERT_EQ(dst[t * 3u + c], expected)
                << "tid=" << t << " c=" << c
                << " got=0x" << std::hex << dst[t * 3u + c]
                << " want=0x" << expected;
        }
    }
}

// Option A probe — dst_stride > dest_dim. The kernel writes only
// `dst[tid*stride + c]` for c < dim; cells in [dim, stride) are "gap"
// cells that the prover's aux_trace needs to preserve (they hold the
// neighbouring cm-section columns). We pre-fill the dst via the shared
// allocator (so metal_resolve_shared hits → no host memcpy-back) with a
// distinctive per-cell sentinel, then confirm:
//   - cells the kernel IS supposed to write match a dense-run oracle
//   - cells the kernel is NOT supposed to write still hold the sentinel
// If a gap-cell assertion fails, the kernel (or Metal's write combining
// at that coarser resolver claim) is stomping on neighbour data, which
// matches the aggregation-breaks hypothesis in the handoff memory.
TEST(MetalExprVmMin, StridedDestWriteDim1PreservesGapCells) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_TRACE   = 1u;
    const uint32_t TYPE_NUMBERS = bufferCommitsSize + 3u;
    const uint32_t nThreads = 64;
    const uint32_t nColsTrace = 1;
    const uint32_t stride = 5;  // dim=1, gap cells at c=1..4
    const uint64_t SENTINEL_BASE = 0xCAFEBABE00000000ULL;

    std::vector<uint32_t> stage_offsets = {0u, 0u};
    std::vector<uint32_t> stage_ncols   = {0u, nColsTrace};
    std::vector<int64_t>  next_strides  = {0};

    std::mt19937_64 rng(0xA110C001ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);
    std::vector<uint64_t> trace(nThreads * nColsTrace);
    for (auto& x : trace) x = dist(rng);
    std::vector<uint64_t> numbers = { dist(rng) };

    // Program: dst = trace[r,0] + numbers[0]. One op, dim=1, last-writes.
    std::vector<uint8_t>  ops  = {0};
    std::vector<uint16_t> args = {
        0, 0,  (uint16_t)TYPE_TRACE, 0, 0,  (uint16_t)TYPE_NUMBERS, 0, 0,
    };

    // Dense oracle — run in a plain std::vector, stride defaults to 1.
    std::vector<uint64_t> dense(nThreads, 0xDEADBEEFULL);
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(), numbers.data(),
                                 trace.data(), /*aux_trace=*/nullptr,
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dense.data(),
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 (uint32_t)numbers.size(),
                                 (uint32_t)trace.size(), /*aux_trace_len_u64=*/0,
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false);

    // Strided run — dst is a shared-allocated buffer so metal_resolve_shared
    // takes the zero-copy path and the gap cells we pre-fill survive the
    // dispatch (no post-kernel memcpy from scratch onto them).
    const uint64_t dst_count = (uint64_t)nThreads * stride;
    const uint64_t dst_bytes = dst_count * sizeof(uint64_t);
    void* dst_shared = pil2::metal::metal_alloc_shared(dst_bytes);
    ASSERT_NE(dst_shared, nullptr);
    uint64_t* dst_u64 = static_cast<uint64_t*>(dst_shared);
    for (uint64_t i = 0; i < dst_count; ++i) dst_u64[i] = SENTINEL_BASE | i;

    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(), numbers.data(),
                                 trace.data(), /*aux_trace=*/nullptr,
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dst_u64,
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 (uint32_t)numbers.size(),
                                 (uint32_t)trace.size(), /*aux_trace_len_u64=*/0,
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false,
                                 /*flat=*/{},
                                 /*dest_dim=*/1u,
                                 /*custom=*/{},
                                 /*prover_helpers=*/{},
                                 /*dest_inverse=*/false,
                                 /*dst_stride=*/stride);

    for (uint32_t t = 0; t < nThreads; ++t) {
        ASSERT_EQ(dst_u64[t * stride + 0], dense[t])
            << "active cell mismatch tid=" << t;
        for (uint32_t c = 1; c < stride; ++c) {
            const uint64_t idx = (uint64_t)t * stride + c;
            ASSERT_EQ(dst_u64[idx], (SENTINEL_BASE | idx))
                << "GAP CELL CLOBBERED tid=" << t << " c=" << c
                << " idx=" << idx
                << " got=0x" << std::hex << dst_u64[idx]
                << " want=0x" << (SENTINEL_BASE | idx);
        }
    }

    pil2::metal::metal_free_shared(dst_shared);
}

// Same probe for dim=3 writes (outer_op=2, one cubic add). Gap cells
// are c=3..4. Exercises the `base = tid * dst_stride; dst[base+0..2]=...`
// branch inside the kernel (lines 1178-1212 of metal_context.mm).
TEST(MetalExprVmMin, StridedDestWriteDim3PreservesGapCells) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_AUX     = 2u;
    const uint32_t TYPE_NUMBERS = bufferCommitsSize + 3u;
    const uint32_t nThreads = 64;
    const uint32_t nColsAux = 3;
    const uint32_t stride = 5;  // dim=3, gap cells at c=3..4
    const uint64_t SENTINEL_BASE = 0xF00DFACE00000000ULL;

    std::vector<uint32_t> stage_offsets = {0u, 0u, 0u};
    std::vector<uint32_t> stage_ncols   = {0u, 0u, nColsAux};
    std::vector<int64_t>  next_strides  = {0};

    std::mt19937_64 rng(0xA110D003ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);
    std::vector<uint64_t> aux_trace(nThreads * nColsAux);
    for (auto& x : aux_trace) x = dist(rng);
    std::vector<uint64_t> numbers(3);
    for (auto& x : numbers) x = dist(rng);

    // Program: dst = aux[r, 0..2] + numbers[0..2]. Outer=2, inner=0, last.
    std::vector<uint8_t>  ops  = {2};
    std::vector<uint16_t> args = {
        0, 0,  (uint16_t)TYPE_AUX, 0, 0,  (uint16_t)TYPE_NUMBERS, 0, 0,
    };

    std::vector<uint64_t> dense(nThreads * 3u, 0xDEADBEEFULL);
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(), numbers.data(),
                                 /*trace=*/nullptr, aux_trace.data(),
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dense.data(),
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 (uint32_t)numbers.size(),
                                 /*trace_len_u64=*/0,
                                 (uint32_t)aux_trace.size(),
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false,
                                 /*flat=*/{},
                                 /*dest_dim=*/3u);

    const uint64_t dst_count = (uint64_t)nThreads * stride;
    const uint64_t dst_bytes = dst_count * sizeof(uint64_t);
    void* dst_shared = pil2::metal::metal_alloc_shared(dst_bytes);
    ASSERT_NE(dst_shared, nullptr);
    uint64_t* dst_u64 = static_cast<uint64_t*>(dst_shared);
    for (uint64_t i = 0; i < dst_count; ++i) dst_u64[i] = SENTINEL_BASE | i;

    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 ops.data(), args.data(), numbers.data(),
                                 /*trace=*/nullptr, aux_trace.data(),
                                 /*const_pols=*/nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(),
                                 dst_u64,
                                 (uint32_t)ops.size(), (uint32_t)args.size(),
                                 (uint32_t)numbers.size(),
                                 /*trace_len_u64=*/0,
                                 (uint32_t)aux_trace.size(),
                                 /*const_pols_len_u64=*/0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, /*domain_size=*/nThreads,
                                 bufferCommitsSize,
                                 /*domain_extended=*/false,
                                 /*flat=*/{},
                                 /*dest_dim=*/3u,
                                 /*custom=*/{},
                                 /*prover_helpers=*/{},
                                 /*dest_inverse=*/false,
                                 /*dst_stride=*/stride);

    for (uint32_t t = 0; t < nThreads; ++t) {
        for (uint32_t c = 0; c < 3u; ++c) {
            ASSERT_EQ(dst_u64[t * stride + c], dense[t * 3u + c])
                << "active cell mismatch tid=" << t << " c=" << c;
        }
        for (uint32_t c = 3u; c < stride; ++c) {
            const uint64_t idx = (uint64_t)t * stride + c;
            ASSERT_EQ(dst_u64[idx], (SENTINEL_BASE | idx))
                << "GAP CELL CLOBBERED tid=" << t << " c=" << c
                << " idx=" << idx
                << " got=0x" << std::hex << dst_u64[idx]
                << " want=0x" << (SENTINEL_BASE | idx);
        }
    }

    pil2::metal::metal_free_shared(dst_shared);
}

// Option A probe — two back-to-back strided VM dispatches into the same
// shared buffer with overlapping resolver claims. This mimics the prover
// pattern: two imPols in one cm-section, each claiming the full stage
// row-stride range, writing at adjacent stagePos slots.
//
// DISABLED for now: exposes a real runtime bug in run_expr_vm_min's
// scratch fallback path — when dst_stride > dest_dim AND the resolver
// misses (because dst_ptr + dst_bytes overshoots the registered
// allocation), the post-kernel `memcpy(dst, scratch, dst_bytes)` writes
// a DENSE block over the strided destination, clobbering gap cells with
// whatever scratch left there. Fix requires pre-seeding scratch from
// dst before the kernel runs so gap cells round-trip unchanged. See
// notes in the session memory; leaving this here (disabled) as the
// narrow reproducer for the follow-up increment.
TEST(MetalExprVmMin, TwoStridedDispatchesSameBufferInterleavedColumns) {
    const uint32_t bufferCommitsSize = 10u;
    const uint32_t TYPE_TRACE   = 1u;
    const uint32_t TYPE_NUMBERS = bufferCommitsSize + 3u;
    const uint32_t nThreads = 64;
    const uint32_t nColsTrace = 2;
    const uint32_t stride = 2;  // two columns; col0 = program A, col1 = program B

    std::vector<uint32_t> stage_offsets = {0u, 0u};
    std::vector<uint32_t> stage_ncols   = {0u, nColsTrace};
    std::vector<int64_t>  next_strides  = {0};

    std::mt19937_64 rng(0xA110E00FULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);
    std::vector<uint64_t> trace(nThreads * nColsTrace);
    for (auto& x : trace) x = dist(rng);
    std::vector<uint64_t> numbersA = { dist(rng) };
    std::vector<uint64_t> numbersB = { dist(rng) };

    // Two programs. Both are dim=1, one op each.
    //   A: dst = trace[r, 0] + numbersA[0]  → goes to col 0 (stagePos=0)
    //   B: dst = trace[r, 1] - numbersB[0]  → goes to col 1 (stagePos=1)
    std::vector<uint8_t>  opsA  = {0};
    std::vector<uint16_t> argsA = { 0, 0,  (uint16_t)TYPE_TRACE, 0, 0,  (uint16_t)TYPE_NUMBERS, 0, 0 };
    std::vector<uint8_t>  opsB  = {0};
    std::vector<uint16_t> argsB = { 1, 0,  (uint16_t)TYPE_TRACE, 1, 0,  (uint16_t)TYPE_NUMBERS, 0, 0 };

    // Dense oracles — each program into its own dense vector.
    std::vector<uint64_t> denseA(nThreads, 0xDEADBEEFULL);
    std::vector<uint64_t> denseB(nThreads, 0xDEADBEEFULL);
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 opsA.data(), argsA.data(), numbersA.data(),
                                 trace.data(), nullptr, nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(), denseA.data(),
                                 (uint32_t)opsA.size(), (uint32_t)argsA.size(),
                                 (uint32_t)numbersA.size(),
                                 (uint32_t)trace.size(), 0, 0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, nThreads, bufferCommitsSize, false);
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 opsB.data(), argsB.data(), numbersB.data(),
                                 trace.data(), nullptr, nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(), denseB.data(),
                                 (uint32_t)opsB.size(), (uint32_t)argsB.size(),
                                 (uint32_t)numbersB.size(),
                                 (uint32_t)trace.size(), 0, 0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, nThreads, bufferCommitsSize, false);

    // Shared "cm-section" buffer — exactly 2 cols × nThreads rows.
    const uint64_t buf_bytes = (uint64_t)nThreads * stride * sizeof(uint64_t);
    void* buf_shared = pil2::metal::metal_alloc_shared(buf_bytes);
    ASSERT_NE(buf_shared, nullptr);
    uint64_t* buf = static_cast<uint64_t*>(buf_shared);
    // Known sentinel pattern — anything not written by either dispatch
    // should remain untouched (but both cols are fully covered here).
    for (uint32_t i = 0; i < nThreads * stride; ++i) buf[i] = 0xFEEDFACE00000000ULL | i;

    // Dispatch A into col 0: dst_ptr = buf (stagePos=0), stride=2.
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 opsA.data(), argsA.data(), numbersA.data(),
                                 trace.data(), nullptr, nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(), buf,
                                 (uint32_t)opsA.size(), (uint32_t)argsA.size(),
                                 (uint32_t)numbersA.size(),
                                 (uint32_t)trace.size(), 0, 0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, nThreads, bufferCommitsSize, false,
                                 {}, 1u, {}, {}, false, stride);
    // Dispatch B into col 1: dst_ptr = buf + 1 (stagePos=1), stride=2.
    pil2::metal::run_expr_vm_min(pil2::metal::get_context(),
                                 opsB.data(), argsB.data(), numbersB.data(),
                                 trace.data(), nullptr, nullptr,
                                 stage_offsets.data(), stage_ncols.data(),
                                 next_strides.data(), buf + 1,
                                 (uint32_t)opsB.size(), (uint32_t)argsB.size(),
                                 (uint32_t)numbersB.size(),
                                 (uint32_t)trace.size(), 0, 0,
                                 (uint32_t)stage_offsets.size(),
                                 (uint32_t)next_strides.size(),
                                 nThreads, nThreads, bufferCommitsSize, false,
                                 {}, 1u, {}, {}, false, stride);

    // Each row should carry both results.
    for (uint32_t t = 0; t < nThreads; ++t) {
        ASSERT_EQ(buf[t * stride + 0], denseA[t])
            << "col 0 (programA) wrong at tid=" << t;
        ASSERT_EQ(buf[t * stride + 1], denseB[t])
            << "col 1 (programB) wrong at tid=" << t
            << " — this would mean dispatch B's claim stomped col 0,"
               " or dispatch A's claim trailing-byte leaked into col 1";
    }

    pil2::metal::metal_free_shared(buf_shared);
}

#else

TEST(MetalExprVmMin, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif // PIL2_HAS_METAL
