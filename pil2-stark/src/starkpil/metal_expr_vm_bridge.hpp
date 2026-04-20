#ifndef PIL2_METAL_EXPR_VM_BRIDGE_HPP
#define PIL2_METAL_EXPR_VM_BRIDGE_HPP

#include "../goldilocks/src/platform.hpp"

#if PIL2_HAS_METAL

#include "../goldilocks/src/metal/metal_context.hpp"
#include "expressions_ctx.hpp"
#include "expressions_bin.hpp"
#include "steps.hpp"
#include "setup_ctx.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Translation layer from the CPU expression-VM state (SetupCtx +
// ExpressionsCtx + StepsParams + Dest) into a single
// pil2::metal::run_expr_vm_min call.
//
// Returns true when the dispatch happened on Metal; the caller should
// skip the CPU path in that case. Returns false on any unsupported
// feature, leaving the output buffer untouched so the CPU fallback can
// run unmodified. Unsupported today (CPU-only fallback):
//   - Multi-dest (dest.params.size() != 1)
//   - Non-tmp dest.params[0] op (cm, const_, number, airvalue)
//   - compilation_time / verify_constraints special paths
//   - Non-power-of-two domain_size (Metal VM uses a bitmask wrap)
//   - Non-natural dest.offset (anything other than 0 or dest.dim)
//   - Stage offset / ncols values that don't fit in u32
//
// Environment gates:
//   PIL2_METAL_EXPR_VM=1       — enable Metal dispatch (CPU path skipped on success)
//   PIL2_METAL_EXPR_VM=0 | off — force CPU path (kill switch)
//   PIL2_METAL_EXPR_VM=verify  — run Metal into a scratch buffer, return false
//                                so CPU still writes dest.dest, then have the
//                                caller diff. Abort on any divergence. Use
//                                this to pinpoint which expression is wrong.
//   PIL2_METAL_EXPR_VM_ALLOW_EXPIDS="7,42,..."  — comma-separated allowlist
//                                of expId values. When set, only listed
//                                expressions route to Metal; others fall
//                                back to CPU. Empty or unset = all allowed.
// Default (unset): Metal dispatch is ON. Consumers built with
// --features metal get the Metal expression-VM automatically (measured
// −20% INNER / −40% FINAL_COMPRESSED with bit-exact tests green).
// To disable at runtime for bisecting: PIL2_METAL_EXPR_VM=0.

namespace pil2 { namespace metal { namespace expr_vm_bridge {

enum class Mode { Off, Run, Verify };

inline Mode metal_expr_vm_mode() {
    static const Mode m = [] {
        const char* e = std::getenv("PIL2_METAL_EXPR_VM");
        // Unset / empty → default ON.
        if (e == nullptr || e[0] == '\0') return Mode::Run;
        // Explicit kill switches: "0" or "off".
        if (e[0] == '0' || std::string(e) == "off") return Mode::Off;
        if (std::string(e) == "verify") return Mode::Verify;
        return Mode::Run;
    }();
    return m;
}

// Comma-separated allowlist from PIL2_METAL_EXPR_VM_ALLOW_EXPIDS. Parsed
// once; empty set means "all expIds allowed".
inline const std::vector<int64_t>& metal_expr_vm_allow_expids() {
    static const std::vector<int64_t> ids = [] {
        std::vector<int64_t> out;
        const char* e = std::getenv("PIL2_METAL_EXPR_VM_ALLOW_EXPIDS");
        if (e == nullptr || e[0] == '\0') return out;
        const char* p = e;
        while (*p) {
            while (*p == ',' || *p == ' ') ++p;
            if (!*p) break;
            char* end = nullptr;
            long long v = std::strtoll(p, &end, 10);
            if (end == p) break;
            out.push_back(static_cast<int64_t>(v));
            p = end;
        }
        return out;
    }();
    return ids;
}

inline bool expid_allowed(int64_t expId) {
    const auto& allow = metal_expr_vm_allow_expids();
    if (allow.empty()) return true;
    for (int64_t a : allow) if (a == expId) return true;
    return false;
}

inline bool is_power_of_two(uint64_t x) {
    return x != 0 && (x & (x - 1ULL)) == 0;
}

// Fill `out` with the first `n_stages_plus_2` entries of a u64 table,
// returning false if any entry exceeds UINT32_MAX. The Metal kernel
// indexes with u32 for now; until widened, bail out on overflow.
inline bool copy_stage_table_u32(const uint64_t* src_u64,
                                 std::vector<uint32_t>& dst_u32,
                                 uint32_t n) {
    dst_u32.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (src_u64[i] > UINT32_MAX) return false;
        dst_u32[i] = static_cast<uint32_t>(src_u64[i]);
    }
    return true;
}

// When `alt_dst != nullptr`, the kernel writes there instead of
// dest.dest. Used by verify mode to keep Metal and CPU outputs separate
// for diffing. Returns true if the dispatch happened.
inline bool try_run_expression_metal(SetupCtx& setupCtx,
                                     ExpressionsCtx& ctx,
                                     StepsParams& params,
                                     Dest& dest,
                                     uint64_t domainSize,
                                     bool domainExtended,
                                     bool compilation_time,
                                     bool verify_constraints,
                                     uint64_t* alt_dst = nullptr) {
    if (metal_expr_vm_mode() == Mode::Off) return false;
    if (compilation_time || verify_constraints) return false;
    // PIL2_METAL_EXPR_VM_DIAG=1 logs every accept/reject decision — run
    // this when tuning the bridge; leaving the env unset keeps it silent.
    const char* diag_env = std::getenv("PIL2_METAL_EXPR_VM_DIAG");
    const bool diag = diag_env && diag_env[0] == '1';
    if (!is_power_of_two(domainSize)) {
        if (diag) std::fprintf(stderr, "[vm-bridge] reject: domainSize=%llu not pow2\n", (unsigned long long)domainSize);
        return false;
    }
    if (dest.params.size() != 1) {
        if (diag) {
            std::fprintf(stderr, "[vm-bridge] reject expId=%lld: dest.params.size=%zu != 1 && != 2 domain=%llu dim=%llu offset=%llu",
                (long long)(dest.params.empty() ? -1 : dest.params[0].expId),
                dest.params.size(), (unsigned long long)domainSize,
                (unsigned long long)dest.dim, (unsigned long long)dest.offset);
            for (size_t k = 0; k < dest.params.size(); ++k) {
                std::fprintf(stderr, " p[%zu].op=%d p[%zu].dim=%llu",
                    k, (int)dest.params[k].op, k,
                    (unsigned long long)dest.params[k].dim);
            }
            std::fprintf(stderr, "\n");
        }
        return false;
    }
    if (dest.params[0].op != opType::tmp) {
        if (diag) std::fprintf(stderr, "[vm-bridge] reject expId=%lld: op=%d != tmp\n", (long long)dest.params[0].expId, (int)dest.params[0].op);
        return false;
    }
    if (dest.dim != 1 && dest.dim != 3) {
        if (diag) std::fprintf(stderr, "[vm-bridge] reject expId=%lld: dim=%llu\n", (long long)dest.params[0].expId, (unsigned long long)dest.dim);
        return false;
    }
    // Strided dest (dest.offset > dim, e.g. imPols writing into a
    // wider cm-section row) falls back to CPU: the Metal strided
    // writeback is 9% slower than the CPU path on fibonacci-square
    // because the resolver-miss path does a per-cell strided memcpy
    // that dominates the kernel win.
    if (dest.offset != 0 && dest.offset != dest.dim) {
        if (diag) std::fprintf(stderr, "[vm-bridge] reject expId=%lld: offset=%llu dim=%llu (strided disabled)\n", (long long)dest.params[0].expId, (unsigned long long)dest.offset, (unsigned long long)dest.dim);
        return false;
    }
    if (!expid_allowed(dest.params[0].expId)) return false;

    // Parser state for this expression.
    ParserArgs& parserArgs = setupCtx.expressionsBin.expressionsBinArgsExpressions;
    auto info_it = setupCtx.expressionsBin.expressionsInfo.find(dest.params[0].expId);
    if (info_it == setupCtx.expressionsBin.expressionsInfo.end()) {
        if (diag) std::fprintf(stderr, "[vm-bridge] reject expId=%lld: not in expressionsInfo\n", (long long)dest.params[0].expId);
        return false;
    }
    ParserParams& parserParams = info_it->second;
    if (parserParams.nOps == 0) {
        if (diag) std::fprintf(stderr, "[vm-bridge] reject expId=%lld: nOps=0\n", (long long)dest.params[0].expId);
        return false;
    }
    // Bytecode scratch must fit the kernel's fixed per-thread pools.
    // Matches EXPR_VM_MAX_TMP1 / EXPR_VM_MAX_TMP3 in metal_context.mm.
    constexpr uint32_t KERNEL_MAX_TMP1 = 128;
    constexpr uint32_t KERNEL_MAX_TMP3 = 32;
    if (parserParams.nTemp1 > KERNEL_MAX_TMP1) {
        if (diag) std::fprintf(stderr, "[vm-bridge] reject expId=%lld: nTemp1=%u > %u\n", (long long)dest.params[0].expId, parserParams.nTemp1, KERNEL_MAX_TMP1);
        return false;
    }
    if (parserParams.nTemp3 > KERNEL_MAX_TMP3) {
        if (diag) std::fprintf(stderr, "[vm-bridge] reject expId=%lld: nTemp3=%u > %u\n", (long long)dest.params[0].expId, parserParams.nTemp3, KERNEL_MAX_TMP3);
        return false;
    }

    // Convert u64 stage tables to u32 (bail if any entry overflows).
    const uint32_t n_stages_plus_2 = static_cast<uint32_t>(1 + ctx.nStages + 1);
    const uint64_t* mapOffsetsUse = domainExtended ? ctx.mapOffsetsExtended
                                                    : ctx.mapOffsets;
    std::vector<uint32_t> stage_offsets;
    std::vector<uint32_t> stage_ncols;
    if (!copy_stage_table_u32(mapOffsetsUse, stage_offsets, n_stages_plus_2)) {
        return false;
    }
    if (!copy_stage_table_u32(ctx.mapSectionsN, stage_ncols, n_stages_plus_2)) {
        return false;
    }

    // Custom commits: same conversion, one entry per custom commit.
    const uint32_t n_custom_commits = static_cast<uint32_t>(setupCtx.starkInfo.customCommits.size());
    std::vector<uint32_t> custom_offsets_u32(n_custom_commits);
    std::vector<uint32_t> custom_ncols_u32(n_custom_commits);
    const uint64_t* customOffsetsUse = domainExtended
        ? ctx.mapOffsetsCustomFixedExtended
        : ctx.mapOffsetsCustomFixed;
    for (uint32_t i = 0; i < n_custom_commits; ++i) {
        if (customOffsetsUse[i] > UINT32_MAX) return false;
        if (ctx.mapSectionsNCustomFixed[i] > UINT32_MAX) return false;
        custom_offsets_u32[i] = static_cast<uint32_t>(customOffsetsUse[i]);
        custom_ncols_u32[i]   = static_cast<uint32_t>(ctx.mapSectionsNCustomFixed[i]);
    }

    // Next-strides: int64 already.
    const int64_t* next_strides = domainExtended ? ctx.nextStridesExtended
                                                  : ctx.nextStrides;
    const uint32_t next_strides_len = static_cast<uint32_t>(setupCtx.starkInfo.openingPoints.size());

    // Buffer lengths (u64 count) — mirror the CPU's own sizing.
    const uint64_t N        = 1ULL << setupCtx.starkInfo.starkStruct.nBits;
    const uint64_t NExt     = 1ULL << setupCtx.starkInfo.starkStruct.nBitsExt;
    const uint32_t nCols_s1 = stage_ncols.size() > 1 ? stage_ncols[1] : 0;
    const uint64_t trace_len_u64 = (params.trace != nullptr && !domainExtended)
                                       ? N * nCols_s1
                                       : 0;
    const uint64_t aux_trace_len_u64 = setupCtx.starkInfo.mapTotalN;
    const uint64_t const_pols_rows   = domainExtended ? NExt : N;
    const uint64_t const_pols_len_u64 = const_pols_rows * setupCtx.starkInfo.nConstants;
    if (trace_len_u64 > UINT32_MAX ||
        aux_trace_len_u64 > UINT32_MAX ||
        const_pols_len_u64 > UINT32_MAX) {
        return false;  // Metal bridge takes u32 lengths today.
    }

    // Flat constant tables (same u64 pointer type on CPU and Metal side).
    ExprVmFlatTables flat;
    flat.public_inputs           = reinterpret_cast<const uint64_t*>(params.publicInputs);
    flat.public_inputs_len_u64   = static_cast<uint32_t>(setupCtx.starkInfo.nPublics);
    flat.air_values              = reinterpret_cast<const uint64_t*>(params.airValues);
    flat.air_values_len_u64      = static_cast<uint32_t>(setupCtx.starkInfo.airValuesSize);
    flat.proof_values            = reinterpret_cast<const uint64_t*>(params.proofValues);
    flat.proof_values_len_u64    = static_cast<uint32_t>(setupCtx.starkInfo.proofValuesSize);
    flat.airgroup_values         = reinterpret_cast<const uint64_t*>(params.airgroupValues);
    flat.airgroup_values_len_u64 = static_cast<uint32_t>(setupCtx.starkInfo.airgroupValuesSize);
    flat.challenges              = reinterpret_cast<const uint64_t*>(params.challenges);
    flat.challenges_len_u64      = static_cast<uint32_t>(ctx.nChallenges) * 3u;
    flat.evals                   = reinterpret_cast<const uint64_t*>(params.evals);
    flat.evals_len_u64           = static_cast<uint32_t>(ctx.nEvals) * 3u;

    // Custom commits struct.
    ExprVmCustomCommits custom;
    if (n_custom_commits > 0) {
        custom.data         = reinterpret_cast<const uint64_t*>(params.pCustomCommitsFixed);
        custom.offsets      = custom_offsets_u32.data();
        custom.ncols        = custom_ncols_u32.data();
        custom.data_len_u64 = static_cast<uint32_t>(setupCtx.starkInfo.mapTotalNCustomCommitsFixed);
        custom.count        = n_custom_commits;
    }

    // Scan the bytecode ONCE up-front to find out which of the
    // bytecode-type sources are actually referenced. Only pass the
    // corresponding buffers to run_expr_vm_min when they are — otherwise
    // we would memcpy from a potentially-uninitialised pointer (ctx.xis
    // is not default-initialised to null; for an AIR that never called
    // setXi it can be garbage, and our fallback memcpy will segfault).
    const uint32_t xi_type   = static_cast<uint32_t>(ctx.bufferCommitsSize - setupCtx.starkInfo.customCommits.size() - 1);
    const uint32_t ph_type   = xi_type - 1u;
    bool bytecode_uses_xi   = false;
    bool bytecode_uses_ph   = false;
    {
        const uint16_t* a = &parserArgs.args[parserParams.argsOffset];
        for (uint32_t k = 0; k < parserParams.nOps; ++k) {
            uint16_t tA = a[k * 8 + 2];
            uint16_t tB = a[k * 8 + 5];
            if (tA == xi_type || tB == xi_type) bytecode_uses_xi = true;
            if (tA == ph_type || tB == ph_type) bytecode_uses_ph = true;
        }
    }

    // Prover helpers: x (extended) / x_n (non-extended) + zi + xis. Each
    // optional; only attach the buffer when the bytecode actually reads it.
    ExprVmProverHelpers ph;
    if (bytecode_uses_ph && ctx.proverHelpers != nullptr) {
        const Goldilocks::Element* x_sel = domainExtended
            ? ctx.proverHelpers->x
            : ctx.proverHelpers->x_n;
        if (x_sel != nullptr) {
            ph.x_current         = reinterpret_cast<const uint64_t*>(x_sel);
            ph.x_current_len_u64 = static_cast<uint32_t>(domainExtended ? NExt : N);
        }
        if (ctx.proverHelpers->zi != nullptr) {
            ph.zi         = reinterpret_cast<const uint64_t*>(ctx.proverHelpers->zi);
            ph.zi_len_u64 = static_cast<uint32_t>(
                setupCtx.starkInfo.boundaries.size() * NExt);
        }
    }
    if (bytecode_uses_xi && ctx.xis != nullptr) {
        ph.xis         = reinterpret_cast<const uint64_t*>(ctx.xis);
        ph.xis_len_u64 = static_cast<uint32_t>(
            setupCtx.starkInfo.openingPoints.size() * 3u);
        // xi compute uses x_current[tid] too — make sure we pass it even
        // if proverHelpers itself is not referenced by the bytecode.
        if (ph.x_current == nullptr && ctx.proverHelpers != nullptr) {
            const Goldilocks::Element* x_sel = domainExtended
                ? ctx.proverHelpers->x
                : ctx.proverHelpers->x_n;
            if (x_sel != nullptr) {
                ph.x_current         = reinterpret_cast<const uint64_t*>(x_sel);
                ph.x_current_len_u64 = static_cast<uint32_t>(domainExtended ? NExt : N);
            }
        }
    }

    const uint32_t n_threads = static_cast<uint32_t>(domainSize);
    const uint32_t domain_size = n_threads;
    const uint32_t dest_dim = static_cast<uint32_t>(dest.dim);

    // The kernel writes `dst[tid * dst_stride + c]`. Pick the stride:
    //   offset == 0 (or == dim) → dense, stride = dest.dim.
    //   offset > dim            → strided write directly into a wider
    //                              row (imPol into cm-section layout).
    // Verify mode writes into a dense scratch from the caller, so use
    // dest.dim there regardless.
    uint32_t dst_stride;
    uint64_t* dst_ptr;
    if (alt_dst != nullptr) {
        dst_ptr = alt_dst;
        dst_stride = static_cast<uint32_t>(dest.dim);
    } else {
        dst_ptr = reinterpret_cast<uint64_t*>(dest.dest);
        dst_stride = (dest.offset == 0) ? static_cast<uint32_t>(dest.dim)
                                        : static_cast<uint32_t>(dest.offset);
    }

    // Diagnostic: in verify mode, summarise which source types this
    // bytecode touches so a divergence report reveals whether xi /
    // custom / prover helpers were involved. Run mode stays silent.
    if (alt_dst != nullptr) {
        const uint32_t xi_type = static_cast<uint32_t>(ctx.bufferCommitsSize - setupCtx.starkInfo.customCommits.size() - 1);
        const uint32_t ph_type = xi_type - 1u;
        const uint16_t* a = &parserArgs.args[parserParams.argsOffset];
        bool uses_xi = false, uses_ph = false, uses_custom = false;
        uint32_t max_outer = 0;
        for (uint32_t k = 0; k < parserParams.nOps; ++k) {
            uint16_t tA = a[k * 8 + 2];
            uint16_t tB = a[k * 8 + 5];
            uint32_t outer = parserArgs.ops[parserParams.opsOffset + k];
            if (outer > max_outer) max_outer = outer;
            auto flag = [&](uint16_t t) {
                if (t == xi_type) uses_xi = true;
                else if (t == ph_type) uses_ph = true;
                else if (t >= (ctx.bufferCommitsSize - setupCtx.starkInfo.customCommits.size())
                         && t < ctx.bufferCommitsSize) uses_custom = true;
            };
            flag(tA); flag(tB);
        }
        std::fprintf(stderr,
            "METAL VM BYTECODE expId=%lld nOps=%u max_outer=%u uses_xi=%d uses_ph=%d uses_custom=%d\n",
            (long long)dest.params[0].expId, parserParams.nOps, max_outer,
            uses_xi ? 1 : 0, uses_ph ? 1 : 0, uses_custom ? 1 : 0);
    }

    pil2::metal::run_expr_vm_min(
        pil2::metal::get_context(),
        &parserArgs.ops[parserParams.opsOffset],
        &parserArgs.args[parserParams.argsOffset],
        reinterpret_cast<const uint64_t*>(parserArgs.numbers),
        reinterpret_cast<const uint64_t*>(params.trace),
        reinterpret_cast<const uint64_t*>(params.aux_trace),
        reinterpret_cast<const uint64_t*>(
            domainExtended ? params.pConstPolsExtendedTreeAddress
                           : params.pConstPolsAddress),
        stage_offsets.data(),
        stage_ncols.data(),
        next_strides,
        dst_ptr,
        parserParams.nOps,
        parserParams.nArgs,
        static_cast<uint32_t>(parserArgs.nNumbers),
        static_cast<uint32_t>(trace_len_u64),
        static_cast<uint32_t>(aux_trace_len_u64),
        static_cast<uint32_t>(const_pols_len_u64),
        n_stages_plus_2,
        next_strides_len,
        n_threads,
        domain_size,
        static_cast<uint32_t>(ctx.bufferCommitsSize),
        domainExtended,
        flat,
        dest_dim,
        custom,
        ph,
        dest.params[0].inverse,
        dst_stride);
    if (diag) std::fprintf(stderr, "[vm-bridge] ACCEPT expId=%lld dim=%llu domain=%llu nOps=%u nTemp1=%u nTemp3=%u stride=%u\n",
                           (long long)dest.params[0].expId, (unsigned long long)dest.dim,
                           (unsigned long long)domainSize, parserParams.nOps,
                           parserParams.nTemp1, parserParams.nTemp3, dst_stride);
    return true;
}

}}}  // namespace pil2::metal::expr_vm_bridge

#endif  // PIL2_HAS_METAL

#endif  // PIL2_METAL_EXPR_VM_BRIDGE_HPP
