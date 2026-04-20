#include "metal_context.hpp"

#if PIL2_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>

namespace pil2::metal {

// Number of persistent scratch slots owned by the context. Each slot is
// lazily allocated and grown to the next power of two on demand. The
// Slot count is chosen to cover the allocation shape of every Metal hot
// path:
//   0..3 — core: LDE/NTT/Merkle/FRI (3 or 4 concurrent buffers each).
//   4..6 — expression VM: pool the three biggest dispatch buffers
//          (aux_trace, const_pols, dst).
//   7..10 — evmap: lev, aux_trace, custom_commits, const_pols_ext.
// `newBufferWithLength` costs ~1.5ms / 32MB, so caching saves alloc
// time across many dispatches. Memcpy still runs on every call — true
// memcpy-free buffer sharing needs a Rust-side MemoryHandler change
// (B.15b proper).
static constexpr int kScratchSlots = 14;

// Maximum number of concurrent Metal streams the scratch pool can
// separate. Each proof-worker thread (set via pil2_metal_set_stream_id
// from the Rust side) borrows from its own scratch[stream][slot] row,
// so two concurrent proofs don't race on the same MTLBuffer. 8 is
// comfortably above the number of streams we're likely to run on
// Apple Silicon; memory cost is (MAX_STREAMS × kScratchSlots) Obj-C
// pointers + size_t's + cache fields, trivial.
static constexpr int kMaxStreams = 8;

enum VmScratchSlot : int {
    kVmSlotAuxTrace   = 4,
    kVmSlotConstPols  = 5,
    kVmSlotDst        = 6,
    kEvmapSlotLev     = 7,
    kEvmapSlotAux     = 8,
    kEvmapSlotCustom  = 9,
    kEvmapSlotConst   = 10,
    // Size=2 expr-VM path: two sub-expression scratches + dst fallback
    // used by run_gl3_mul when the caller's pointers aren't registered.
    kGl3MulSrcA       = 11,
    kGl3MulSrcB       = 12,
    kGl3MulDst        = 13,
};

struct Context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;        // Legacy single-queue handle. Stream 0
                                       // uses this; code paths that don't
                                       // care about streams read through
                                       // get_queue() → queues[0].
    id<MTLCommandQueue> queues[kMaxStreams];
    // Persistent scratch pool — amortises Metal's newBufferWithLength cost
    // (~1.5 ms per 32 MB) across all calls of a given shape. Each slot
    // stores its last-allocated size so callers get the existing buffer
    // when their request fits, or a larger power-of-two-rounded buffer
    // when they outgrow it. Reads/writes serialised by `scratch_mutex`.
    //
    // Two-dimensional per-stream: scratch[stream_id][slot]. Each Metal-
    // calling thread picks its stream via the thread-local set by
    // pil2_metal_set_stream_id (Rust sets it when spawning proof
    // workers). Without this separation, two concurrent proofs would
    // race on the same `MTLBuffer` when `scratch_borrow(slot)` returns
    // the same handle — shared auxiliary tables (roots, Poseidon
    // constants) then get half-written by the other stream, corrupting
    // later kernels. Aggregation's Merkle re-verify catches that bug;
    // plain `prove` doesn't because both prover and verifier see the
    // same corruption. Stream 0 is the default (main thread).
    id<MTLBuffer> scratch[kMaxStreams][kScratchSlots];
    size_t        scratch_bytes[kMaxStreams][kScratchSlots];
    // Per-slot memcpy cache. When a caller asks to upload (src_ptr, bytes)
    // with matching first+last u64 signatures, we can skip the memcpy —
    // buffer contents are still valid. Used for write-once inputs like
    // const_pols. The signature check catches the fast-deallocation
    // pattern where an earlier vector is freed and a new vector with
    // different content lands at the same virtual address. Bulletproof
    // against intentional cache staleness it's not, but callers using
    // scratch_upload_cached should be feeding data that's stable for
    // the reuse window anyway. Per-stream too (const_pols stream 0
    // sees the same content on every call, stream 1 also sees its own
    // cache — they don't interfere).
    const void*   scratch_src_ptr[kMaxStreams][kScratchSlots];
    size_t        scratch_src_bytes[kMaxStreams][kScratchSlots];
    uint64_t      scratch_src_sig_first[kMaxStreams][kScratchSlots];
    uint64_t      scratch_src_sig_last[kMaxStreams][kScratchSlots];
    std::mutex    scratch_mutex;
    // MSL compile cache + PSO cache. `newLibraryWithSource` is cheap on
    // cache hits (shader cache handles source hashing), but
    // `newComputePipelineStateWithFunction` does real compilation work
    // (~few ms per kernel). Since every hot-path function re-creates
    // the PSO per dispatch, caching by kernel name shaves ms × many
    // calls per proof. Single library instance, map of name → PSO.
    id<MTLLibrary> cached_library;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pso_cache;
    std::mutex pso_mutex;
    // Unified-memory registry (B.18). Buffers allocated via
    // metal_alloc_shared are tracked here so the VM bridge (and later
    // other kernels) can bind them zero-copy instead of memcpy-ing
    // through a scratch slot. The key is the `.contents` pointer the
    // allocator handed out; `size_bytes` lets range-lookup resolve
    // caller pointers that sit inside a registered allocation
    // (e.g. dst = aux_trace + offset_f).
    struct SharedEntry {
        id<MTLBuffer> buffer;
        size_t        size_bytes;
    };
    std::unordered_map<void*, SharedEntry> shared_buffers;
    std::mutex shared_mutex;
};

// Current thread's Metal stream id. Set by pil2_metal_set_stream_id
// (called by the Rust side when it spawns a proof worker). Defaults to
// 0 so single-threaded callers — main thread, tests, witness
// computation that happens to invoke Metal — land on a consistent slot
// row. Out of anonymous namespace so set_current_stream_id below can
// take its address; the static storage is still private to this TU.
static thread_local int t_stream_id = 0;

[[noreturn]] static void stream_id_fatal(const char* msg) {
    std::fprintf(stderr, "pil2::metal fatal: %s\n", msg);
    std::abort();
}

void set_current_stream_id(int id) {
    if (id < 0 || id >= kMaxStreams) { stream_id_fatal("set_current_stream_id: out of range"); }
    t_stream_id = id;
}

namespace {

std::once_flag g_init_flag;
Context* g_ctx = nullptr;

int current_stream_id() {
    int s = t_stream_id;
    if (s < 0 || s >= kMaxStreams) { stream_id_fatal("stream_id out of range"); }
    return s;
}

[[noreturn]] void fatal(const char* msg) {
    std::fprintf(stderr, "pil2::metal fatal: %s\n", msg);
    std::abort();
}

// Forward declaration of the MSL source blob (defined below near
// get_or_make_pso). Needed for the PSO warmup call inside
// init_context.
extern const char* kMetalSrc;

void init_context() {
    // Register the stats dump before Metal setup so it fires at process
    // exit regardless of where init failed. No-op when PIL2_METAL_STATS
    // is unset.
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fatal("MTLCreateSystemDefaultDevice returned nil"); }
        auto* c = new Context();
        c->device = dev;
        // One MTLCommandQueue per stream. Sharing a single queue across
        // concurrent proof workers can interleave command-buffer
        // submissions in ways the validation layer tolerates but that
        // (empirically) produce wrong Merkle-tree bytes — aggregation's
        // recursion circuit catches the mismatch. Separate queues let
        // Metal schedule each stream's work independently and is the
        // thread-safe pattern Apple recommends for concurrent producers.
        for (int s = 0; s < kMaxStreams; ++s) {
            id<MTLCommandQueue> q = [dev newCommandQueue];
            if (!q) { fatal("newCommandQueue returned nil"); }
            c->queues[s] = q;
        }
        c->queue = c->queues[0];
        for (int s = 0; s < kMaxStreams; ++s) {
            for (int i = 0; i < kScratchSlots; ++i) {
                c->scratch[s][i] = nil;
                c->scratch_bytes[s][i] = 0;
                c->scratch_src_ptr[s][i] = nullptr;
                c->scratch_src_bytes[s][i] = 0;
                c->scratch_src_sig_first[s][i] = 0;
                c->scratch_src_sig_last[s][i]  = 0;
            }
        }
        c->cached_library = nil;

        // Eagerly compile the MSL library and build a PSO for every
        // kernel the prover calls. Amortises the ~100-500ms per-kernel
        // PSO creation cost out of the first prove path (where cold
        // compilation previously stacked on top of the first real
        // dispatch — fibonacci-square's first recursive1 was 3.3s vs
        // ~1.2s for later ones, most of which traces to this).
        //
        // Explicit list: deriving it from MSL text would need parsing.
        // If a new kernel lands and isn't added here, the worst case
        // is a single first-call penalty on that kernel — not a
        // correctness issue.
        {
            NSError* err = nil;
            NSString* src = [NSString stringWithUTF8String:kMetalSrc];
            id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
            if (!lib) {
                NSString* msg = err ? [err localizedDescription] : @"(no error info)";
                std::fprintf(stderr, "pil2::metal: warmup MSL compile failed: %s\n",
                             [msg UTF8String]);
                std::abort();
            }
            c->cached_library = lib;
            static const char* kWarmupKernels[] = {
                // Field ops (scalar + cubic)
                "field_add_k", "field_sub_k", "field_mul_k",
                "gl3_add_k", "gl3_sub_k", "gl3_mul_k", "gl3_mul_scalar_k",
                "gl3_mul_strided_k",
                "gl_inv_k", "gl3_inv_k",
                "gl_op_k", "gl3_op_k", "gl3_op_31_k",
                // NTT / LDE
                "ntt_reverse_permutation", "ntt_butterfly_phase",
                "ntt_radix4_phase", "ntt_radix8_phase",
                "ntt_rev_butterfly_s1s2s3",
                "intt_reorder_scale", "intt_reorder_coset_scale",
                // Merkle / Poseidon2
                "pose2_compress_w8", "pose2_compress_w12", "pose2_compress_w16",
                "pose2_leaf_hash_w8", "pose2_leaf_hash_w12", "pose2_leaf_hash_w16",
                "pose2_linear_hash_w8", "pose2_linear_hash_w12", "pose2_linear_hash_w16",
                "poseidon2_permute_w8_batch", "poseidon2_permute_w12_batch", "poseidon2_permute_w16_batch",
                // FRI / evmap / expression VM
                "fri_fold_w8_k", "evmap_k", "expr_vm_min_k",
                // Misc
                "noop_write",
            };
            for (const char* name : kWarmupKernels) {
                NSString* nm = [NSString stringWithUTF8String:name];
                id<MTLFunction> fn = [lib newFunctionWithName:nm];
                if (!fn) continue;  // kernel renamed/removed: skip silently
                NSError* perr = nil;
                id<MTLComputePipelineState> pso =
                    [dev newComputePipelineStateWithFunction:fn error:&perr];
                if (pso) c->pso_cache[std::string(name)] = pso;
                // On failure fall back silently — the real get_or_make_pso
                // path will abort with a clear error if the kernel is
                // actually needed later.
            }
        }

        g_ctx = c;
    }
}

// Borrow a persistent MTLBuffer of at least `bytes` from the context's
// scratch pool. The buffer is owned by the context; callers must not
// release it. If the slot's existing buffer is too small (or nil), it is
// replaced by a freshly allocated buffer rounded up to the next power of
// two — ARC releases the previous one when the last strong reference
// drops. Returned contents are undefined; caller is responsible for
// initialising the regions it reads.
//
// Shape convention across callers (by slot index, within a single Metal
// dispatch function which is serial under the shared command queue):
//   slot 0 — primary work/output buffer (bufExt in lde_metal,
//            bufOut in poseidon2_permute, bufTree in merkletree)
//   slot 1 — input-from-host buffer     (bufRoots in lde_metal,
//            bufIn in poseidon2 / merkletree)
//   slot 2 — first parameter table      (bufR in lde_metal,
//            bufC in poseidon2 / merkletree)
//   slot 3 — second parameter table     (bufD in poseidon2 / merkletree)
id<MTLBuffer> scratch_borrow(Context* ctx, int slot, size_t bytes) {
    if (slot < 0 || slot >= kScratchSlots) { fatal("scratch_borrow: bad slot"); }
    if (bytes == 0) bytes = 1;
    const int stream = current_stream_id();
    std::lock_guard<std::mutex> lock(ctx->scratch_mutex);
    if (ctx->scratch[stream][slot] == nil || ctx->scratch_bytes[stream][slot] < bytes) {
        size_t alloc_bytes = 1;
        while (alloc_bytes < bytes) alloc_bytes <<= 1;
        id<MTLBuffer> buf =
            [ctx->device newBufferWithLength:alloc_bytes
                                     options:MTLResourceStorageModeShared];
        if (!buf) { fatal("scratch_borrow: alloc failed"); }
        ctx->scratch[stream][slot] = buf;
        ctx->scratch_bytes[stream][slot] = alloc_bytes;
        // Grow invalidates the memcpy cache for this slot.
        ctx->scratch_src_ptr[stream][slot] = nullptr;
        ctx->scratch_src_bytes[stream][slot] = 0;
        ctx->scratch_src_sig_first[stream][slot] = 0;
        ctx->scratch_src_sig_last[stream][slot]  = 0;
    }
    return ctx->scratch[stream][slot];
}

// Borrow + upload with cache. Skips the memcpy when the same src_ptr
// also matches the first and last u64 signatures from the prior upload.
// This catches the case where a caller's buffer was freed and a new
// one landed at the same virtual address with different content — the
// signature check will diverge. Callers should still only use this on
// data that is logically stable between uses (e.g. const_pols).
id<MTLBuffer> scratch_upload_cached(Context* ctx, int slot,
                                    const void* src, size_t bytes) {
    id<MTLBuffer> buf = scratch_borrow(ctx, slot, bytes);
    // Compute signature from first + last u64 (or 1 byte if tiny).
    uint64_t sig_first = 0, sig_last = 0;
    if (bytes >= sizeof(uint64_t)) {
        sig_first = *static_cast<const uint64_t*>(src);
        sig_last  = *reinterpret_cast<const uint64_t*>(
            static_cast<const uint8_t*>(src) + bytes - sizeof(uint64_t));
    } else if (bytes > 0) {
        sig_first = *static_cast<const uint8_t*>(src);
        sig_last  = sig_first;
    }
    const int stream = current_stream_id();
    std::lock_guard<std::mutex> lock(ctx->scratch_mutex);
    const bool hit = ctx->scratch_src_ptr[stream][slot]   == src
                  && ctx->scratch_src_bytes[stream][slot] == bytes
                  && ctx->scratch_src_sig_first[stream][slot] == sig_first
                  && ctx->scratch_src_sig_last[stream][slot]  == sig_last;
    if (!hit) {
        std::memcpy([buf contents], src, bytes);
        ctx->scratch_src_ptr[stream][slot]       = src;
        ctx->scratch_src_bytes[stream][slot]     = bytes;
        ctx->scratch_src_sig_first[stream][slot] = sig_first;
        ctx->scratch_src_sig_last[stream][slot]  = sig_last;
    }
    return buf;
}

// Resolve a caller pointer (+ bytes) to a registered MTLBuffer, returning
// the byte offset within that buffer. False if the range doesn't fit
// entirely inside a single registered allocation. Internal — the VM
// bridge and future kernels use this as the first step in buffer setup.
bool metal_resolve_shared(Context* ctx,
                          const void* ptr, size_t bytes,
                          id<MTLBuffer>* out_buffer,
                          size_t* out_offset) {
    if (!ctx || ptr == nullptr) { return false; }
    std::lock_guard<std::mutex> lock(ctx->shared_mutex);
    if (ctx->shared_buffers.empty()) { return false; }
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    for (const auto& kv : ctx->shared_buffers) {
        const uint8_t* base = static_cast<const uint8_t*>(kv.first);
        const size_t sz = kv.second.size_bytes;
        if (p >= base && (p + bytes) <= (base + sz)) {
            *out_buffer = kv.second.buffer;
            *out_offset = static_cast<size_t>(p - base);
            return true;
        }
    }
    return false;
}

// Diagnostic helper: returns true if `ptr` lies inside a registered
// shared buffer (regardless of whether `bytes` fits). Fills the
// allocation base / size / in-allocation-offset so callers can compute
// why a resolve failed. Used by the STRIDED_MISS diag.
bool metal_shared_containing(Context* ctx,
                             const void* ptr,
                             const void** out_base,
                             size_t* out_size,
                             size_t* out_offset) {
    if (!ctx || ptr == nullptr) return false;
    std::lock_guard<std::mutex> lock(ctx->shared_mutex);
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    for (const auto& kv : ctx->shared_buffers) {
        const uint8_t* base = static_cast<const uint8_t*>(kv.first);
        const size_t sz = kv.second.size_bytes;
        if (p >= base && p < base + sz) {
            if (out_base)   *out_base   = base;
            if (out_size)   *out_size   = sz;
            if (out_offset) *out_offset = static_cast<size_t>(p - base);
            return true;
        }
    }
    return false;
}

} // namespace

void* metal_alloc_shared(uint64_t bytes) {
    if (bytes == 0) return nullptr;
    Context* ctx = g_ctx;
    std::call_once(g_init_flag, init_context);
    ctx = g_ctx;
    if (!ctx) fatal("metal_alloc_shared: context init failed");
    id<MTLBuffer> buf = [ctx->device newBufferWithLength:static_cast<NSUInteger>(bytes)
                                                 options:MTLResourceStorageModeShared];
    if (!buf) fatal("metal_alloc_shared: newBufferWithLength failed");
    void* p = [buf contents];
    {
        std::lock_guard<std::mutex> lock(ctx->shared_mutex);
        Context::SharedEntry entry{ buf, static_cast<size_t>(bytes) };
        ctx->shared_buffers[p] = entry;
    }
    return p;
}

void metal_free_shared(void* ptr) {
    if (ptr == nullptr) return;
    Context* ctx = get_context();
    std::lock_guard<std::mutex> lock(ctx->shared_mutex);
    auto it = ctx->shared_buffers.find(ptr);
    if (it == ctx->shared_buffers.end()) {
        std::fprintf(stderr,
            "pil2::metal::metal_free_shared: pointer %p not found in registry\n",
            ptr);
        return;
    }
    // ARC releases `it->second.buffer` when the map entry is erased.
    const size_t freed_bytes = it->second.size_bytes;
    ctx->shared_buffers.erase(it);
}

bool metal_is_shared_base(const void* ptr) {
    if (ptr == nullptr) return false;
    Context* ctx = g_ctx;
    if (!ctx) return false;
    std::lock_guard<std::mutex> lock(ctx->shared_mutex);
    return ctx->shared_buffers.find(const_cast<void*>(ptr)) != ctx->shared_buffers.end();
}

} // namespace pil2::metal

// C linkage FFI wrappers for Rust / non-C++ callers. Declared here rather
// than in metal_context.hpp to keep the Rust FFI independent of C++
// name mangling / namespaces.
extern "C" {

void* pil2_metal_alloc_shared(uint64_t bytes) {
    return pil2::metal::metal_alloc_shared(bytes);
}

void pil2_metal_free_shared(void* ptr) {
    pil2::metal::metal_free_shared(ptr);
}

int pil2_metal_is_shared_base(const void* ptr) {
    return pil2::metal::metal_is_shared_base(ptr) ? 1 : 0;
}

// Set the current thread's Metal stream id. Subsequent scratch_borrow /
// scratch_upload_cached calls on this thread hit scratch[stream][slot]
// instead of scratch[0][slot], so two concurrent proof workers don't
// race on the same auxiliary buffer. Must be called once per proof-
// worker thread before any Metal call. `id` must be in [0, 8).
void pil2_metal_set_stream_id(int id) {
    pil2::metal::set_current_stream_id(id);
}

} // extern "C"

namespace pil2::metal {

ContextHandle get_context() {
    std::call_once(g_init_flag, init_context);
    return g_ctx;
}

std::string device_name(ContextHandle ctx) {
    if (!ctx) { return {}; }
    NSString* name = [ctx->device name];
    const char* utf = [name UTF8String];
    return std::string(utf ? utf : "");
}

namespace {

// Embedded MSL for the C3 smoke test. Kept in the .mm (not a standalone
// .metal + AOT metallib) so the runtime has no file-path dependency — the
// test has to pass whether it's run from the goldilocks dir or a different
// cwd once the library is linked into a consumer binary.
const char* kNoopSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void noop_write(device uint* out [[buffer(0)]],
                       uint tid         [[thread_position_in_grid]]) {
    if (tid == 0) {
        out[0] = 42u;
    }
}
)MSL";

} // namespace

uint32_t run_noop_test(ContextHandle ctx) {
    if (!ctx) { fatal("run_noop_test: null context"); }
    @autoreleasepool {
        id<MTLDevice>       dev = ctx->device;
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];

        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:kNoopSource];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            NSString* msg = err ? [err localizedDescription] : @"(no error info)";
            std::fprintf(stderr, "pil2::metal: MSL compile failed: %s\n", [msg UTF8String]);
            std::abort();
        }

        id<MTLFunction> fn = [lib newFunctionWithName:@"noop_write"];
        if (!fn) { fatal("newFunctionWithName(noop_write) returned nil"); }

        err = nil;
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            NSString* msg = err ? [err localizedDescription] : @"(no error info)";
            std::fprintf(stderr, "pil2::metal: PSO create failed: %s\n", [msg UTF8String]);
            std::abort();
        }

        id<MTLBuffer> buf = [dev newBufferWithLength:sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
        if (!buf) { fatal("newBufferWithLength failed"); }
        *static_cast<uint32_t*>([buf contents]) = 0u;

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: command buffer error: %s\n", [msg UTF8String]);
            std::abort();
        }

        return *static_cast<uint32_t*>([buf contents]);
    }
}

// --------------------------------------------------------------------------
// Phase C4-C5: Goldilocks field arithmetic + NTT kernels (MSL)
// --------------------------------------------------------------------------
//
// Direct port of ~/Development/goldilocks/src/metal/kernels/{field,ntt}.metal.
// Same prime (2^64 - 2^32 + 1), same lazy-reduce contract:
//   gl_add, gl_sub      -> fully reduced in [0, p)
//   gl_mul              -> lazy reduced in [0, 2p)
//   gl_canonicalize     -> [0, 2p) -> [0, p), call at kernel exit
//
// Kept here as an embedded raw string literal (not a standalone .metal file
// on disk) so the runtime has no file-path dependency. When we add more
// kernels (Poseidon2, Merkle) and the MSL grows large enough that embedding
// is awkward, we'll extract into .metal files + build-time codegen. For now
// one copy, one place. The C++ helpers compile this shared source and look
// up the kernel they need by name; Metal's shader cache dedups the compile
// across calls with identical source.
namespace {

const char* kMetalSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant ulong GL_PRIME = 0xFFFFFFFF00000001UL;
constant ulong GL_CQ    = 0xFFFFFFFFUL;

// Bug fix vs. the ~/Development/goldilocks reference: the carry handling
// below corrects s when a+b overflows u64, but NOT when a+b stays under
// 2^64 yet lands in [P, 2^64) — most notably a+b == P exactly (equivalence
// class 0), which is reachable e.g. with a=1, b=P-1. Without the final
// reduction the kernel returns P, CPU returns 0, bit-exact test fails.
// Fix is one branch-free subtract; cost is negligible.
inline ulong gl_add(ulong a, ulong b) {
    ulong s = a + b;
    ulong carry = (s < a) ? 1UL : 0UL;
    s += carry * GL_CQ;
    ulong carry2 = (carry != 0 && s < GL_CQ) ? 1UL : 0UL;
    s += carry2 * GL_CQ;
    if (s >= GL_PRIME) s -= GL_PRIME;
    return s;
}

inline ulong gl_sub(ulong a, ulong b) {
    ulong s = a - b;
    ulong borrow = (s > a) ? 1UL : 0UL;
    ulong prev = s;
    s -= borrow * GL_CQ;
    ulong borrow2 = (borrow != 0 && s > prev) ? 1UL : 0UL;
    s -= borrow2 * GL_CQ;
    return s;
}

inline ulong gl_mul(ulong a, ulong b) {
    // 3-multiplication Goldilocks reduction (Plonky2 style).
    //
    // Derivation. For the 128-bit product c = c_hi * 2^64 + c_lo,
    // split c_hi = hi_hi * 2^32 + hi_lo. Using 2^64 ≡ 2^32-1 (mod p)
    // twice:
    //   c mod p = c_lo + hi_lo * 2^32 - hi_hi - hi_lo
    //           = c_lo + hi_lo * GL_CQ - hi_hi
    //           = (c_lo - hi_hi) + hi_lo * GL_CQ
    // The prior implementation computed the same expression as three
    // sequential add/subs with explicit carry3/adj-mul bookkeeping
    // (4 muls total). This form needs only TWO final corrections --
    // one borrow on (c_lo - hi_hi), one carry on the final add --
    // for 3 multiplication-class ops total (mul + mulhi + hi_lo*GL_CQ).
    ulong c_lo  = a * b;
    ulong c_hi  = metal::mulhi(a, b);
    ulong hi_hi = c_hi >> 32;
    ulong hi_lo = c_hi & GL_CQ;

    // t0 = c_lo - hi_hi; if underflow, subtract GL_CQ (fold 2^64 ≡ GL_CQ).
    ulong t0 = c_lo - hi_hi;
    t0 -= (t0 > c_lo) ? GL_CQ : 0UL;

    // hi_lo * GL_CQ fits in u64 (both < 2^32).
    ulong t1 = hi_lo * GL_CQ;

    // t2 = t0 + t1; if overflow, add GL_CQ.
    ulong t2 = t0 + t1;
    t2 += (t2 < t0) ? GL_CQ : 0UL;

    return t2;  // lazy: u64 ≡ result mod p
}

inline ulong gl_canonicalize(ulong a) {
    return (a >= GL_PRIME) ? (a - GL_PRIME) : a;
}

// Multiply u64 x by a small constant k (k must fit in u32), reducing mod
// p. Uses only 32-bit widening multiplications, skipping the full 128-bit
// product reduction that gl_mul needs. Roughly half the ops of gl_mul for
// k-values in [2, 2^32). Output is lazy (u64 ≡ result mod p, not
// canonicalised).
//
// Port of the same primitive from ~/Development/goldilocks. Useful in
// Poseidon2 matmul where coefficients are small (2, 4, 6, 7 for the M4
// MDS matrix). Compared to expressing k*x as a chain of gl_adds, this
// shifts work from the Integer-and-Conditional pipeline (carry chains)
// to the Integer-and-Complex pipeline (widening muls) — the GPU
// profile shows Conditional saturates at 83% limiter while Complex
// has headroom at 62%.
inline ulong gl_mul_small(ulong x, uint k) {
    uint  x_lo = (uint)(x & 0xFFFFFFFFu);
    uint  x_hi = (uint)(x >> 32);
    ulong p_lo = (ulong)x_lo * (ulong)k;   // widening 32×32 → 64
    ulong p_hi = (ulong)x_hi * (ulong)k;

    // 96-bit product = (p_hi << 32) + p_lo, split into (hi_32, lo_64).
    ulong lo_64 = p_lo + (p_hi << 32);
    uint  carry = (lo_64 < p_lo) ? 1u : 0u;
    uint  hi_32 = (uint)(p_hi >> 32) + carry;

    // Reduce: 2^64 ≡ GL_CQ (mod p). product mod p ≡ lo_64 + hi_32*GL_CQ.
    // hi_32 < 2^32 and GL_CQ < 2^32, so hi_32*GL_CQ fits in u64.
    ulong hi_cq  = (ulong)hi_32 * GL_CQ;
    ulong result = lo_64 + hi_cq;
    result += (result < lo_64) ? GL_CQ : 0UL;  // single-step overflow fixup
    return result;                              // lazy [0, 2p)
}

// --- Goldilocks cubic extension F_p[x] / (x^3 - x - 1) --------------------
// Mirrors Goldilocks3::{add,sub,mul} from
// pil2-stark/src/goldilocks/src/goldilocks_cubic_extension.hpp. Element layout
// is 3 consecutive ulong values [c0, c1, c2] representing c0 + c1*x + c2*x^2.
// gl_mul returns lazy [0, 2p), so multiplication products are canonicalised
// before being fed back into gl_add/gl_sub (which require canonical inputs
// post-patch to produce canonical outputs). Inputs to gl_mul don't need to
// be canonical — the algorithm tolerates [0, 2^64).
inline void gl3_add(thread ulong       (&out)[3],
                    thread const ulong (&a)[3],
                    thread const ulong (&b)[3]) {
    out[0] = gl_add(a[0], b[0]);
    out[1] = gl_add(a[1], b[1]);
    out[2] = gl_add(a[2], b[2]);
}

inline void gl3_sub(thread ulong       (&out)[3],
                    thread const ulong (&a)[3],
                    thread const ulong (&b)[3]) {
    out[0] = gl_sub(a[0], b[0]);
    out[1] = gl_sub(a[1], b[1]);
    out[2] = gl_sub(a[2], b[2]);
}

// 6-base-field-mul Karatsuba-style product for x^3 - x - 1. Matches
// Goldilocks3::mul (goldilocks_cubic_extension.hpp:232-245) exactly.
inline void gl3_mul(thread ulong       (&out)[3],
                    thread const ulong (&a)[3],
                    thread const ulong (&b)[3]) {
    const ulong a01 = gl_add(a[0], a[1]);
    const ulong a02 = gl_add(a[0], a[2]);
    const ulong a12 = gl_add(a[1], a[2]);
    const ulong b01 = gl_add(b[0], b[1]);
    const ulong b02 = gl_add(b[0], b[2]);
    const ulong b12 = gl_add(b[1], b[2]);
    const ulong A = gl_canonicalize(gl_mul(a01,  b01));
    const ulong B = gl_canonicalize(gl_mul(a02,  b02));
    const ulong C = gl_canonicalize(gl_mul(a12,  b12));
    const ulong D = gl_canonicalize(gl_mul(a[0], b[0]));
    const ulong E = gl_canonicalize(gl_mul(a[1], b[1]));
    const ulong F = gl_canonicalize(gl_mul(a[2], b[2]));
    const ulong G = gl_sub(D, E);
    out[0] = gl_sub(gl_add(C, G), F);
    out[1] = gl_sub(gl_sub(gl_sub(gl_add(A, C), E), E), D);
    out[2] = gl_sub(B, G);
}

// Scalar · cubic-ext: out = a * s where s is a base-field scalar.
inline void gl3_mul_scalar(thread ulong       (&out)[3],
                           thread const ulong (&a)[3],
                           ulong s) {
    out[0] = gl_canonicalize(gl_mul(a[0], s));
    out[1] = gl_canonicalize(gl_mul(a[1], s));
    out[2] = gl_canonicalize(gl_mul(a[2], s));
}

// Base-field inverse via Fermat's little theorem: a^(p-2) mod p.
// p - 2 == 0xFFFFFFFEFFFFFFFF (63 meaningful bits). Square-and-multiply,
// LSB-first. Caller contract: a != 0 (a == 0 returns 0 here but CPU
// Goldilocks::inv throws; no ambiguity for the expression VM because
// gl3_inv is only invoked on non-zero t by construction).
inline ulong gl_inv(ulong a) {
    const ulong EXP = 0xFFFFFFFEFFFFFFFFUL;
    ulong acc  = 1UL;
    ulong base = gl_canonicalize(a);
    ulong e    = EXP;
    for (uint i = 0; i < 64u; ++i) {
        if ((e & 1UL) != 0UL) {
            acc = gl_canonicalize(gl_mul(acc, base));
        }
        base = gl_canonicalize(gl_mul(base, base));
        e >>= 1;
    }
    return acc;
}

// Cubic-extension inverse in F_p[x] / (x^3 - x - 1). Direct port of
// Goldilocks3::inv (goldilocks_cubic_extension.hpp:289-318): 14 base-
// field products, one base-field inverse, a handful of adds/subs.
// Output canonical.
inline void gl3_inv(thread ulong       (&result)[3],
                    thread const ulong (&a)[3]) {
    ulong a0 = gl_canonicalize(a[0]);
    ulong a1 = gl_canonicalize(a[1]);
    ulong a2 = gl_canonicalize(a[2]);

    ulong aa = gl_canonicalize(gl_mul(a0, a0));
    ulong ac = gl_canonicalize(gl_mul(a0, a2));
    ulong ba = gl_canonicalize(gl_mul(a1, a0));
    ulong bb = gl_canonicalize(gl_mul(a1, a1));
    ulong bc = gl_canonicalize(gl_mul(a1, a2));
    ulong cc = gl_canonicalize(gl_mul(a2, a2));

    ulong aaa = gl_canonicalize(gl_mul(aa, a0));
    ulong aac = gl_canonicalize(gl_mul(aa, a2));
    ulong abc = gl_canonicalize(gl_mul(ba, a2));
    ulong abb = gl_canonicalize(gl_mul(ba, a1));
    ulong acc_ = gl_canonicalize(gl_mul(ac, a2));
    ulong bbb = gl_canonicalize(gl_mul(bb, a1));
    ulong bcc = gl_canonicalize(gl_mul(bc, a2));
    ulong ccc = gl_canonicalize(gl_mul(cc, a2));

    // t = 3*abc + abb - aaa - 2*aac - acc - bbb + bcc - ccc
    ulong t = gl_add(abc, abc);
    t = gl_add(t, abc);
    t = gl_add(t, abb);
    t = gl_sub(t, aaa);
    t = gl_sub(t, aac);
    t = gl_sub(t, aac);
    t = gl_sub(t, acc_);
    t = gl_sub(t, bbb);
    t = gl_add(t, bcc);
    t = gl_sub(t, ccc);

    ulong tinv = gl_inv(t);

    ulong s1 = gl_add(bc, bb);
    s1 = gl_sub(s1, aa);
    s1 = gl_sub(s1, ac);
    s1 = gl_sub(s1, ac);
    s1 = gl_sub(s1, cc);
    result[0] = gl_canonicalize(gl_mul(s1, tinv));

    ulong s2 = gl_sub(ba, cc);
    result[1] = gl_canonicalize(gl_mul(s2, tinv));

    ulong s3 = gl_add(ac, cc);
    s3 = gl_sub(s3, bb);
    result[2] = gl_canonicalize(gl_mul(s3, tinv));
}

// --- op-code dispatchers --------------------------------------------------
// Mirror Goldilocks::op_pack, Goldilocks3::op_pack, Goldilocks3::op_31_pack
// from the CPU expressions VM. Op encoding:
//   0 → c = a + b
//   1 → c = a - b
//   2 → c = a * b  (canonicalised for base field)
//   3 → c = b - a  (reverse subtract)
// These feed a future expression-VM port; the VM dispatches on op codes
// loaded from the bytecode stream.

// Base field. Inputs assumed canonical; output canonical.
inline ulong gl_op(uint op, ulong a, ulong b) {
    switch (op) {
        case 0u: return gl_add(a, b);
        case 1u: return gl_sub(a, b);
        case 2u: return gl_canonicalize(gl_mul(a, b));
        default: return gl_sub(b, a);  // case 3
    }
}

// Cubic ext · cubic ext.
inline void gl3_op(uint op,
                   thread ulong       (&out)[3],
                   thread const ulong (&a)[3],
                   thread const ulong (&b)[3]) {
    switch (op) {
        case 0u: gl3_add(out, a, b);     return;
        case 1u: gl3_sub(out, a, b);     return;
        case 2u: gl3_mul(out, a, b);     return;
        default: gl3_sub(out, b, a);     return;  // case 3
    }
}

// Cubic ext · base field. Matches Goldilocks3::op_31_pack: ops 0/1 only
// touch the first lane and pass through the upper lanes; op 2 is scalar
// multiplication; op 3 negates the upper lanes (b − a in cubic-ext sense).
inline void gl3_op_31(uint op,
                      thread ulong       (&out)[3],
                      thread const ulong (&a)[3],
                      ulong b) {
    switch (op) {
        case 0u:
            out[0] = gl_add(a[0], b);
            out[1] = a[1];
            out[2] = a[2];
            return;
        case 1u:
            out[0] = gl_sub(a[0], b);
            out[1] = a[1];
            out[2] = a[2];
            return;
        case 2u:
            gl3_mul_scalar(out, a, b);
            return;
        default:  // case 3: c = b − a  (first lane regular sub; upper lanes negated)
            out[0] = gl_sub(b, a[0]);
            out[1] = gl_sub(0UL, a[1]);
            out[2] = gl_sub(0UL, a[2]);
            return;
    }
}

// --- Phase C4 test kernels: pairwise op over two arrays ----------------
// Each kernel reads a[tid], b[tid], writes canonical result to out[tid].
// `n` is passed as a setBytes constant (avoids a dedicated buffer for one
// u32). `gl_mul` gets a canonicalize so the output is directly comparable
// to the CPU reference's [0, p) convention.

kernel void field_add_k(const device ulong* a [[buffer(0)]],
                        const device ulong* b [[buffer(1)]],
                        device       ulong* o [[buffer(2)]],
                        constant uint&      n [[buffer(3)]],
                        uint tid              [[thread_position_in_grid]]) {
    if (tid >= n) return;
    o[tid] = gl_add(a[tid], b[tid]);
}

kernel void field_sub_k(const device ulong* a [[buffer(0)]],
                        const device ulong* b [[buffer(1)]],
                        device       ulong* o [[buffer(2)]],
                        constant uint&      n [[buffer(3)]],
                        uint tid              [[thread_position_in_grid]]) {
    if (tid >= n) return;
    o[tid] = gl_sub(a[tid], b[tid]);
}

kernel void field_mul_k(const device ulong* a [[buffer(0)]],
                        const device ulong* b [[buffer(1)]],
                        device       ulong* o [[buffer(2)]],
                        constant uint&      n [[buffer(3)]],
                        uint tid              [[thread_position_in_grid]]) {
    if (tid >= n) return;
    o[tid] = gl_canonicalize(gl_mul(a[tid], b[tid]));
}

// Cubic-extension test kernels. Each thread consumes 3 u64s from `a` and
// `b` (one gl3 element each), writes 3 u64s to `o`. `op` selects the
// operation via the kernel name rather than a runtime branch, so the
// compiler inlines the single call. Layout: out[tid*3 + c] = op(a, b)[c].
kernel void gl3_add_k(const device ulong* a [[buffer(0)]],
                      const device ulong* b [[buffer(1)]],
                      device       ulong* o [[buffer(2)]],
                      constant uint&      n [[buffer(3)]],
                      uint tid              [[thread_position_in_grid]]) {
    if (tid >= n) return;
    thread ulong aa[3] = { a[tid*3+0], a[tid*3+1], a[tid*3+2] };
    thread ulong bb[3] = { b[tid*3+0], b[tid*3+1], b[tid*3+2] };
    thread ulong rr[3];
    gl3_add(rr, aa, bb);
    o[tid*3+0] = rr[0]; o[tid*3+1] = rr[1]; o[tid*3+2] = rr[2];
}

kernel void gl3_sub_k(const device ulong* a [[buffer(0)]],
                      const device ulong* b [[buffer(1)]],
                      device       ulong* o [[buffer(2)]],
                      constant uint&      n [[buffer(3)]],
                      uint tid              [[thread_position_in_grid]]) {
    if (tid >= n) return;
    thread ulong aa[3] = { a[tid*3+0], a[tid*3+1], a[tid*3+2] };
    thread ulong bb[3] = { b[tid*3+0], b[tid*3+1], b[tid*3+2] };
    thread ulong rr[3];
    gl3_sub(rr, aa, bb);
    o[tid*3+0] = rr[0]; o[tid*3+1] = rr[1]; o[tid*3+2] = rr[2];
}

kernel void gl3_mul_k(const device ulong* a [[buffer(0)]],
                      const device ulong* b [[buffer(1)]],
                      device       ulong* o [[buffer(2)]],
                      constant uint&      n [[buffer(3)]],
                      uint tid              [[thread_position_in_grid]]) {
    if (tid >= n) return;
    thread ulong aa[3] = { a[tid*3+0], a[tid*3+1], a[tid*3+2] };
    thread ulong bb[3] = { b[tid*3+0], b[tid*3+1], b[tid*3+2] };
    thread ulong rr[3];
    gl3_mul(rr, aa, bb);
    o[tid*3+0] = rr[0]; o[tid*3+1] = rr[1]; o[tid*3+2] = rr[2];
}

// Strided-write variant: srcs stay dense (stride 3), dst writes at
// tid * dst_stride + c for c in [0, 3). Used by the size=2 expr-VM path
// when the outer dest.offset > dest.dim (imPols writing into a wider
// cm-section row). Separate from gl3_mul_k so the dense hot path stays
// uniform-count minimal.
kernel void gl3_mul_strided_k(const device ulong* a          [[buffer(0)]],
                              const device ulong* b          [[buffer(1)]],
                              device       ulong* o          [[buffer(2)]],
                              constant uint&      n          [[buffer(3)]],
                              constant uint&      dst_stride [[buffer(4)]],
                              uint tid                       [[thread_position_in_grid]]) {
    if (tid >= n) return;
    thread ulong aa[3] = { a[tid*3+0], a[tid*3+1], a[tid*3+2] };
    thread ulong bb[3] = { b[tid*3+0], b[tid*3+1], b[tid*3+2] };
    thread ulong rr[3];
    gl3_mul(rr, aa, bb);
    const uint base = tid * dst_stride;
    o[base + 0] = rr[0]; o[base + 1] = rr[1]; o[base + 2] = rr[2];
}

// out[i] = a[i] * s[i/3]: scalar lives one-per-element in `b`.
kernel void gl3_mul_scalar_k(const device ulong* a [[buffer(0)]],
                             const device ulong* b [[buffer(1)]],
                             device       ulong* o [[buffer(2)]],
                             constant uint&      n [[buffer(3)]],
                             uint tid              [[thread_position_in_grid]]) {
    if (tid >= n) return;
    thread ulong aa[3] = { a[tid*3+0], a[tid*3+1], a[tid*3+2] };
    thread ulong rr[3];
    gl3_mul_scalar(rr, aa, b[tid]);
    o[tid*3+0] = rr[0]; o[tid*3+1] = rr[1]; o[tid*3+2] = rr[2];
}

// Base-field Fermat inverse test kernel — one element per thread.
kernel void gl_inv_k(const device ulong* a [[buffer(0)]],
                     device       ulong* o [[buffer(1)]],
                     constant uint&      n [[buffer(2)]],
                     uint tid              [[thread_position_in_grid]]) {
    if (tid >= n) return;
    o[tid] = gl_inv(a[tid]);
}

// Cubic-extension inverse test kernel — 3 u64s per thread in/out.
kernel void gl3_inv_k(const device ulong* a [[buffer(0)]],
                      device       ulong* o [[buffer(1)]],
                      constant uint&      n [[buffer(2)]],
                      uint tid              [[thread_position_in_grid]]) {
    if (tid >= n) return;
    thread ulong aa[3] = { a[tid*3+0], a[tid*3+1], a[tid*3+2] };
    thread ulong rr[3];
    gl3_inv(rr, aa);
    o[tid*3+0] = rr[0]; o[tid*3+1] = rr[1]; o[tid*3+2] = rr[2];
}

// Op-dispatcher test kernels. `op` is a runtime constant that drives the
// 4-way switch inside the device helper. Inputs/outputs identical to the
// non-dispatch variants.
kernel void gl_op_k(const device ulong* a   [[buffer(0)]],
                    const device ulong* b   [[buffer(1)]],
                    device       ulong* o   [[buffer(2)]],
                    constant uint&      n   [[buffer(3)]],
                    constant uint&      op  [[buffer(4)]],
                    uint tid                [[thread_position_in_grid]]) {
    if (tid >= n) return;
    o[tid] = gl_op(op, a[tid], b[tid]);
}

kernel void gl3_op_k(const device ulong* a   [[buffer(0)]],
                     const device ulong* b   [[buffer(1)]],
                     device       ulong* o   [[buffer(2)]],
                     constant uint&      n   [[buffer(3)]],
                     constant uint&      op  [[buffer(4)]],
                     uint tid                [[thread_position_in_grid]]) {
    if (tid >= n) return;
    thread ulong aa[3] = { a[tid*3+0], a[tid*3+1], a[tid*3+2] };
    thread ulong bb[3] = { b[tid*3+0], b[tid*3+1], b[tid*3+2] };
    thread ulong rr[3];
    gl3_op(op, rr, aa, bb);
    o[tid*3+0] = rr[0]; o[tid*3+1] = rr[1]; o[tid*3+2] = rr[2];
}

kernel void gl3_op_31_k(const device ulong* a   [[buffer(0)]],
                        const device ulong* b   [[buffer(1)]],
                        device       ulong* o   [[buffer(2)]],
                        constant uint&      n   [[buffer(3)]],
                        constant uint&      op  [[buffer(4)]],
                        uint tid                [[thread_position_in_grid]]) {
    if (tid >= n) return;
    thread ulong aa[3] = { a[tid*3+0], a[tid*3+1], a[tid*3+2] };
    thread ulong rr[3];
    gl3_op_31(op, rr, aa, b[tid]);
    o[tid*3+0] = rr[0]; o[tid*3+1] = rr[1]; o[tid*3+2] = rr[2];
}

// --- Expression-VM minimum-viable kernel (Phase E, Steps B.0..B.11) ----
// Per-thread interpreter for a straight-line bytecode program. Scope grows
// one source type per step so each kernel revision stays verifiable.
// Current coverage:
//   - Outer op 0 (dim1 × dim1 → dim1) — gl_op
//   - Outer op 1 (dim3 × dim1 → dim3) — gl3_op_31
//   - Outer op 2 (dim3 × dim3 → dim3) — gl3_op
//   - Source types newly added in B.9:
//       * type in [bCS - n_custom_commits, bCS - 1] : customCommits
//         (per-commit offset + row*ncols + slot; dim-1 on CPU, widened
//         defensively for cubic reads).
//   - Source types newly added in B.10:
//       * type == customCommitsLoBound - 2 (= nStages + 2): proverHelpers
//         on the prover path. slot (= boundary) == 0 → x_current[row]
//         (caller supplies x when domain_extended, else x_n). slot > 0 →
//         zi[(slot - 1) * domain_size + row]. Both dim-1.
//   - Source type newly added in B.11:
//       * type == customCommitsLoBound - 1 (= nStages + 3): xi (cubic
//         opening-point reciprocal). Prover compute-on-read — uses the
//         gl3_inv primitive from B.11a. Reads:
//           diff = (x_current[row] - xis[slot*3], -xis[slot*3+1],
//                   -xis[slot*3+2])                       // cubic
//           out  = gl3_inv(diff)                          // cubic
//         CPU asserts dim==3 here (cubic only); expr_vm_load returns 0
//         defensively for the dim-1 path.
// All 13 source types now wired. Semantic gaps that remain before full
// integration: post-op cubic batch inverse (dest.inverse), multi-dest
// multiplication (dest.params.size() == 2).
//   - Source types (all 11 base-field paths still valid; cubic reads
//     handled by expr_vm_load3):
//       * type == 0                        : const_pols  (dim 1 only)
//       * type == 1, !domainExtended       : trace       (dim 1 only)
//       * type == 1,  domainExtended       : aux_trace@stage_offsets[1]
//       * type in [2, nStages+1]           : aux_trace per-stage
//       * type == bufferCommitsSize        : tmp1        (dim 1)
//       * type == bufferCommitsSize + 1    : tmp3        (dim 3)
//       * type == bufferCommitsSize + 2..8 : flat constant tables
//   - tmp1 holds up to EXPR_VM_MAX_TMP1 slots per thread (dim-1 scratch).
//   - tmp3 holds up to EXPR_VM_MAX_TMP3 cubic slots per thread.
// Inner op codes 0..3 (add, sub, mul, reverse-sub) go through gl_op /
// gl3_op depending on outer-op dim.
constant uint EXPR_VM_MAX_TMP1 = 128;
constant uint EXPR_VM_MAX_TMP3 = 32;

// `tid` is the per-thread unmodified row index; `row` is `(tid + offset) & mask`
// from the bytecode's rowOffsetIdx. const_pols / trace / aux_trace / custom
// commits all apply the row offset and take `row`. proverHelpers and xi on the
// CPU side (expressions_pack.hpp) read directly from the outer loop's `i`
// without any offset, so they use `tid` here.
inline ulong expr_vm_load(uint type, uint slot, uint tid, uint row,
                          thread const ulong (&tmp1)[EXPR_VM_MAX_TMP1],
                          device const ulong* const_pols,
                          device const ulong* trace,
                          device const ulong* aux_trace,
                          device const ulong* numbers,
                          device const ulong* public_inputs,
                          device const ulong* air_values,
                          device const ulong* proof_values,
                          device const ulong* airgroup_values,
                          device const ulong* challenges,
                          device const ulong* evals,
                          device const ulong* custom_commits,
                          device const uint*  custom_offsets,
                          device const uint*  custom_ncols,
                          device const ulong* x_current,
                          device const ulong* zi,
                          device const ulong* xis,
                          device const uint*  stage_offsets,
                          device const uint*  stage_ncols,
                          uint bufferCommitsSize,
                          uint customCommitsLoBound,
                          uint domainSize,
                          bool domainExtended) {
    if (type == 0u) {
        return const_pols[stage_offsets[0] + row * stage_ncols[0] + slot];
    }
    // Only the raw trace buffer is stage-1-specific to the non-extended
    // domain; on the extended domain stage 1 lives inside aux_trace like
    // every other stage, so we fall through to the aux_trace branch below.
    if (type == 1u && !domainExtended) {
        return trace[row * stage_ncols[1] + slot];
    }
    if (type == bufferCommitsSize)        return tmp1[slot];
    // proverHelpers: type == nStages+2 == customCommitsLoBound - 2.
    // CPU reads proverHelpers->x / x_n / zi at the unmodified outer-loop
    // row — rowOffsetIndex is intentionally ignored. Use `tid`, not the
    // offset-wrapped `row`.
    if (type + 2u == customCommitsLoBound) {
        if (slot == 0u) return x_current[tid];
        return zi[(slot - 1u) * domainSize + tid];
    }
    // xi (type == nStages+3). CPU asserts dim==3 — the dim-1 path here
    // would mean a bytecode bug. Return 0 defensively rather than reading
    // uninitialised memory. (void) to silence unused param warnings.
    if (type + 1u == customCommitsLoBound) { (void)xis; return 0UL; }
    if (type == bufferCommitsSize + 2u)   return public_inputs[slot];
    if (type == bufferCommitsSize + 3u)   return numbers[slot];
    if (type == bufferCommitsSize + 4u)   return air_values[slot];
    if (type == bufferCommitsSize + 5u)   return proof_values[slot];
    if (type == bufferCommitsSize + 6u)   return airgroup_values[slot];
    if (type == bufferCommitsSize + 7u)   return challenges[slot];
    if (type == bufferCommitsSize + 8u)   return evals[slot];
    // customCommits: indexed range [customCommitsLoBound, bufferCommitsSize).
    // Check before aux_trace fallthrough so custom-commit types never leak
    // into the aux_trace index math.
    if (type >= customCommitsLoBound && type < bufferCommitsSize) {
        uint idx = type - customCommitsLoBound;
        return custom_commits[custom_offsets[idx] + row * custom_ncols[idx] + slot];
    }
    // Fallthrough: aux_trace. Caller is responsible for emitting only
    // types in [1, nStages+1] here when domainExtended, or [2, nStages+1]
    // otherwise — any other type is a bytecode bug.
    return aux_trace[stage_offsets[type] + row * stage_ncols[type] + slot];
}

// Cubic (dim=3) loader — reads 3 u64s into `out`. Mirrors expr_vm_load's
// type routing. Flat sources read 3 consecutive slots (the standard
// bytecode layout for cubic constants). aux_trace reads 3 contiguous
// u64s at the row's cubic column. tmp3 reads from per-thread scratch.
// Dim-1 sources (const_pols, trace, tmp1) shouldn't be called with
// dim=3 in well-formed bytecode — we widen to [val, 0, 0] defensively.
inline void expr_vm_load3(thread ulong (&out)[3],
                          uint type, uint slot, uint tid, uint row,
                          thread const ulong (&tmp1)[EXPR_VM_MAX_TMP1],
                          thread const ulong (&tmp3)[EXPR_VM_MAX_TMP3 * 3u],
                          device const ulong* const_pols,
                          device const ulong* trace,
                          device const ulong* aux_trace,
                          device const ulong* numbers,
                          device const ulong* public_inputs,
                          device const ulong* air_values,
                          device const ulong* proof_values,
                          device const ulong* airgroup_values,
                          device const ulong* challenges,
                          device const ulong* evals,
                          device const ulong* custom_commits,
                          device const uint*  custom_offsets,
                          device const uint*  custom_ncols,
                          device const ulong* x_current,
                          device const ulong* zi,
                          device const ulong* xis,
                          device const uint*  stage_offsets,
                          device const uint*  stage_ncols,
                          uint bufferCommitsSize,
                          uint customCommitsLoBound,
                          uint domainSize,
                          bool domainExtended) {
    // xi (type == nStages+3): compute-on-read. diff = (x_tid, 0, 0) - xi
    // as cubic, then cubic inverse. CPU reads proverHelpers->x[tid] with
    // the UNMODIFIED outer-row index, so use `tid` not `row` here (the
    // rowOffsetIndex on the xi operand is ignored by the CPU's load()).
    if (type + 1u == customCommitsLoBound) {
        thread ulong diff[3];
        diff[0] = gl_sub(x_current[tid], xis[slot * 3u + 0u]);
        diff[1] = gl_sub(0UL,            xis[slot * 3u + 1u]);
        diff[2] = gl_sub(0UL,            xis[slot * 3u + 2u]);
        gl3_inv(out, diff);
        return;
    }
    if (type == bufferCommitsSize + 1u) {
        out[0] = tmp3[slot * 3u + 0u];
        out[1] = tmp3[slot * 3u + 1u];
        out[2] = tmp3[slot * 3u + 2u];
        return;
    }
    // Flat constant tables: 3 consecutive u64s starting at `slot`.
    if (type == bufferCommitsSize + 2u) { out[0] = public_inputs[slot];   out[1] = public_inputs[slot + 1u];   out[2] = public_inputs[slot + 2u];   return; }
    if (type == bufferCommitsSize + 3u) { out[0] = numbers[slot];         out[1] = numbers[slot + 1u];         out[2] = numbers[slot + 2u];         return; }
    if (type == bufferCommitsSize + 4u) { out[0] = air_values[slot];      out[1] = air_values[slot + 1u];      out[2] = air_values[slot + 2u];      return; }
    if (type == bufferCommitsSize + 5u) { out[0] = proof_values[slot];    out[1] = proof_values[slot + 1u];    out[2] = proof_values[slot + 2u];    return; }
    if (type == bufferCommitsSize + 6u) { out[0] = airgroup_values[slot]; out[1] = airgroup_values[slot + 1u]; out[2] = airgroup_values[slot + 2u]; return; }
    if (type == bufferCommitsSize + 7u) { out[0] = challenges[slot];      out[1] = challenges[slot + 1u];      out[2] = challenges[slot + 2u];      return; }
    if (type == bufferCommitsSize + 8u) { out[0] = evals[slot];           out[1] = evals[slot + 1u];           out[2] = evals[slot + 2u];           return; }

    // Dim-1 sources reached with dim=3 — bytecode bug on CPU exits(-1);
    // here we widen defensively so misuse produces obviously-wrong but
    // non-crashing output.
    if (type == 0u) {
        out[0] = const_pols[stage_offsets[0] + row * stage_ncols[0] + slot];
        out[1] = 0UL; out[2] = 0UL;
        return;
    }
    if (type == 1u && !domainExtended) {
        out[0] = trace[row * stage_ncols[1] + slot];
        out[1] = 0UL; out[2] = 0UL;
        return;
    }
    if (type == bufferCommitsSize) {
        out[0] = tmp1[slot]; out[1] = 0UL; out[2] = 0UL;
        return;
    }
    // proverHelpers (dim-1). Widen defensively. Uses tid not row (see
    // load's proverHelpers branch for rationale).
    if (type + 2u == customCommitsLoBound) {
        ulong v = (slot == 0u) ? x_current[tid] : zi[(slot - 1u) * domainSize + tid];
        out[0] = v; out[1] = 0UL; out[2] = 0UL;
        return;
    }
    // customCommits: dim-1 on CPU. Widen defensively.
    if (type >= customCommitsLoBound && type < bufferCommitsSize) {
        uint idx = type - customCommitsLoBound;
        out[0] = custom_commits[custom_offsets[idx] + row * custom_ncols[idx] + slot];
        out[1] = 0UL; out[2] = 0UL;
        return;
    }
    // Fallthrough: aux_trace cubic width — 3 contiguous u64s at the
    // row's cubic column.
    uint base = stage_offsets[type] + row * stage_ncols[type] + slot;
    out[0] = aux_trace[base + 0u];
    out[1] = aux_trace[base + 1u];
    out[2] = aux_trace[base + 2u];
}

kernel void expr_vm_min_k(
    device const uchar*  ops               [[buffer(0)]],
    device const ushort* args              [[buffer(1)]],
    device const ulong*  numbers           [[buffer(2)]],
    device       ulong*  dst               [[buffer(3)]],
    constant uint&       nOps              [[buffer(4)]],
    constant uint&       nThreads          [[buffer(5)]],
    constant uint&       bufferCommitsSize [[buffer(6)]],
    device const ulong*  trace             [[buffer(7)]],
    device const ulong*  aux_trace         [[buffer(8)]],
    device const uint*   stage_offsets     [[buffer(9)]],
    device const uint*   stage_ncols       [[buffer(10)]],
    constant uchar&      domainExtended    [[buffer(11)]],
    device const ulong*  const_pols        [[buffer(12)]],
    device const ulong*  public_inputs     [[buffer(13)]],
    device const ulong*  air_values        [[buffer(14)]],
    device const ulong*  proof_values      [[buffer(15)]],
    device const ulong*  airgroup_values   [[buffer(16)]],
    device const ulong*  challenges        [[buffer(17)]],
    device const ulong*  evals             [[buffer(18)]],
    device const long*   next_strides      [[buffer(19)]],
    constant uint&       domainSize        [[buffer(20)]],
    device const ulong*  custom_commits    [[buffer(21)]],
    device const uint*   custom_offsets    [[buffer(22)]],
    device const uint*   custom_ncols      [[buffer(23)]],
    constant uint&       customCommitsLoBound [[buffer(24)]],
    device const ulong*  x_current         [[buffer(25)]],
    device const ulong*  zi                [[buffer(26)]],
    device const ulong*  xis               [[buffer(27)]],
    constant uint&       dst_stride        [[buffer(28)]],
    uint tid                               [[thread_position_in_grid]])
{
    if (tid >= nThreads) return;

    thread ulong tmp1[EXPR_VM_MAX_TMP1];
    thread ulong tmp3[EXPR_VM_MAX_TMP3 * 3u];
    bool de = (domainExtended != 0u);
    // domainSize is asserted pow-of-2 by the host; mask == wrap-around
    // modulo the domain. Unsigned bitmask on a signed sum still yields
    // the correct cyclic index for negative strides because two's-
    // complement preserves the low bits of (tid + offset) modulo 2^64.
    ulong dom_mask = (ulong)(domainSize - 1u);

    uint i_args = 0u;
    for (uint kk = 0u; kk < nOps; ++kk) {
        uint outer          = (uint)ops[kk];
        uint inner_op       = (uint)args[i_args + 0];
        uint dest_slot      = (uint)args[i_args + 1];
        uint typeA          = (uint)args[i_args + 2];
        uint slotA          = (uint)args[i_args + 3];
        uint rowOffsetIdxA  = (uint)args[i_args + 4];
        uint typeB          = (uint)args[i_args + 5];
        uint slotB          = (uint)args[i_args + 6];
        uint rowOffsetIdxB  = (uint)args[i_args + 7];

        long offA = next_strides[rowOffsetIdxA];
        long offB = next_strides[rowOffsetIdxB];
        uint rowA = (uint)(((long)tid + offA) & (long)dom_mask);
        uint rowB = (uint)(((long)tid + offB) & (long)dom_mask);

        bool is_last = (kk + 1u == nOps);

        if (outer == 0u) {
            ulong a = expr_vm_load(typeA, slotA, tid, rowA, tmp1,
                                   const_pols, trace, aux_trace, numbers,
                                   public_inputs, air_values, proof_values,
                                   airgroup_values, challenges, evals,
                                   custom_commits, custom_offsets, custom_ncols,
                                   x_current, zi, xis,
                                   stage_offsets, stage_ncols, bufferCommitsSize,
                                   customCommitsLoBound, domainSize, de);
            ulong b = expr_vm_load(typeB, slotB, tid, rowB, tmp1,
                                   const_pols, trace, aux_trace, numbers,
                                   public_inputs, air_values, proof_values,
                                   airgroup_values, challenges, evals,
                                   custom_commits, custom_offsets, custom_ncols,
                                   x_current, zi, xis,
                                   stage_offsets, stage_ncols, bufferCommitsSize,
                                   customCommitsLoBound, domainSize, de);
            ulong r = gl_op(inner_op, a, b);
            if (is_last) dst[tid * dst_stride] = r;
            else         tmp1[dest_slot] = r;
        } else if (outer == 1u) {
            thread ulong aa[3];
            thread ulong rr[3];
            expr_vm_load3(aa, typeA, slotA, tid, rowA, tmp1, tmp3,
                          const_pols, trace, aux_trace, numbers,
                          public_inputs, air_values, proof_values,
                          airgroup_values, challenges, evals,
                          custom_commits, custom_offsets, custom_ncols,
                          x_current, zi, xis,
                          stage_offsets, stage_ncols, bufferCommitsSize,
                          customCommitsLoBound, domainSize, de);
            ulong b = expr_vm_load(typeB, slotB, tid, rowB, tmp1,
                                   const_pols, trace, aux_trace, numbers,
                                   public_inputs, air_values, proof_values,
                                   airgroup_values, challenges, evals,
                                   custom_commits, custom_offsets, custom_ncols,
                                   x_current, zi, xis,
                                   stage_offsets, stage_ncols, bufferCommitsSize,
                                   customCommitsLoBound, domainSize, de);
            gl3_op_31(inner_op, rr, aa, b);
            if (is_last) {
                uint base = tid * dst_stride;
                dst[base + 0u] = rr[0];
                dst[base + 1u] = rr[1];
                dst[base + 2u] = rr[2];
            } else {
                tmp3[dest_slot * 3u + 0u] = rr[0];
                tmp3[dest_slot * 3u + 1u] = rr[1];
                tmp3[dest_slot * 3u + 2u] = rr[2];
            }
        } else if (outer == 2u) {
            thread ulong aa[3];
            thread ulong bb[3];
            thread ulong rr[3];
            expr_vm_load3(aa, typeA, slotA, tid, rowA, tmp1, tmp3,
                          const_pols, trace, aux_trace, numbers,
                          public_inputs, air_values, proof_values,
                          airgroup_values, challenges, evals,
                          custom_commits, custom_offsets, custom_ncols,
                          x_current, zi, xis,
                          stage_offsets, stage_ncols, bufferCommitsSize,
                          customCommitsLoBound, domainSize, de);
            expr_vm_load3(bb, typeB, slotB, tid, rowB, tmp1, tmp3,
                          const_pols, trace, aux_trace, numbers,
                          public_inputs, air_values, proof_values,
                          airgroup_values, challenges, evals,
                          custom_commits, custom_offsets, custom_ncols,
                          x_current, zi, xis,
                          stage_offsets, stage_ncols, bufferCommitsSize,
                          customCommitsLoBound, domainSize, de);
            gl3_op(inner_op, rr, aa, bb);
            if (is_last) {
                uint base = tid * dst_stride;
                dst[base + 0u] = rr[0];
                dst[base + 1u] = rr[1];
                dst[base + 2u] = rr[2];
            } else {
                tmp3[dest_slot * 3u + 0u] = rr[0];
                tmp3[dest_slot * 3u + 1u] = rr[1];
                tmp3[dest_slot * 3u + 2u] = rr[2];
            }
        }

        i_args += 8u;
    }
}

// --- Phase C5a: NTT minimal path (rev-perm + radix-2 butterfly) --------

inline uint reverse_bits32(uint x) {
    x = ((x >> 1)  & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2)  & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4)  & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
    x = ((x >> 8)  & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8);
    x = (x >> 16) | (x << 16);
    return x;
}

// In-place bit-reversal permutation on a row-major (size x ncols) buffer.
// Each thread handles one row index `tid`; it swaps row tid with row rev(tid)
// only when rev > tid, so each pair is touched exactly once.
kernel void ntt_reverse_permutation(
    device ulong*   buf        [[buffer(0)]],
    constant uint&  domain_pow [[buffer(1)]],
    constant uint&  ncols      [[buffer(2)]],
    uint tid                   [[thread_position_in_grid]])
{
    uint domain_size = 1u << domain_pow;
    if (tid >= domain_size) return;
    uint rev = reverse_bits32(tid) >> (32u - domain_pow);
    if (rev <= tid) return;
    uint src = tid * ncols;
    uint dst = rev * ncols;
    for (uint c = 0; c < ncols; c++) {
        ulong tmp = buf[src + c];
        buf[src + c] = buf[dst + c];
        buf[dst + c] = tmp;
    }
}

// One radix-2 Cooley-Tukey butterfly phase. Stage `s` processes butterflies
// of length m = 2^s; each butterfly at offset `(group*m + j)` pairs with
// `(group*m + j + m/2)` and uses twiddle w^j where w is the primitive m-th
// root of unity.
//
// The `twiddles` buffer holds the full (roots_len)-th roots of unity; to
// fetch the twiddle for stage `s`, index `j << roots_stride_shift` where
// `roots_stride_shift = log2(roots_len) - s`. This lets a single roots
// buffer serve every phase.
//
// gl_canonicalize after gl_mul keeps `t` in [0, p) so the gl_add/gl_sub
// preconditions hold even through chained phases.
kernel void ntt_butterfly_phase(
    device ulong*       buf                [[buffer(0)]],
    device const ulong* twiddles           [[buffer(1)]],
    constant uint&      ncols              [[buffer(2)]],
    constant uint&      domain_size        [[buffer(3)]],
    constant uint&      s                  [[buffer(4)]],
    constant uint&      roots_stride_shift [[buffer(5)]],
    uint tid                               [[thread_position_in_grid]])
{
    uint half_n = domain_size >> 1;
    uint total  = half_n * ncols;
    if (tid >= total) return;

    uint pair_idx = tid / ncols;
    uint col      = tid % ncols;
    uint m        = 1u << s;
    uint mdiv2    = m >> 1;
    uint group    = pair_idx / mdiv2;
    uint j        = pair_idx % mdiv2;

    uint off2 = (group * m + j) * ncols + col;
    uint off1 = off2 + mdiv2 * ncols;

    ulong w = twiddles[(uint)j << roots_stride_shift];
    ulong u = buf[off2];
    ulong t = gl_canonicalize(gl_mul(w, buf[off1]));
    buf[off2] = gl_add(u, t);
    buf[off1] = gl_sub(u, t);
}

// Radix-4 phase: collapses two consecutive Cooley-Tukey DIT stages (s, s+1)
// into one kernel dispatch, halving global-memory round-trips at the cost of
// more register pressure per thread. Each thread reads 4 elements spaced
// M/4 apart (where M = 2^(s+1)) and writes them back in place.
//
// Algorithm (equivalent to back-to-back radix-2 at s then s+1):
//   inner stage s:   y_A = (x_A0 ± w_s * x_A1),  y_B = (x_B0 ± w_s * x_B1)
//   outer stage s+1: z_0 = y_A0 + w_q   * y_B0
//                    z_1 = y_A1 + w_qI  * y_B1
//                    z_2 = y_A0 - w_q   * y_B0
//                    z_3 = y_A1 - w_qI  * y_B1
//
// Twiddles: `stride_s1 = roots_pow - (s+1)`. Inner twiddle w_s uses stride
// `stride_s1 + 1` (equivalent to `roots_pow - s`); outer twiddles use
// `stride_s1` directly. This is the same full roots buffer as radix-2.
//
// No explicit canonicalize after gl_mul: our gl_add / gl_sub reduce any
// u64 input correctly (see the "bug fix" comment above gl_add), so the
// lazy gl_mul output can flow straight into the adds/subs.
kernel void ntt_radix4_phase(
    device ulong*       buf         [[buffer(0)]],
    device const ulong* twiddles    [[buffer(1)]],
    constant uint&      ncols       [[buffer(2)]],
    constant uint&      domain_size [[buffer(3)]],
    constant uint&      s           [[buffer(4)]],
    constant uint&      stride_s1   [[buffer(5)]],
    uint tid                        [[thread_position_in_grid]])
{
    uint quarter_n = domain_size >> 2;
    uint total     = quarter_n * ncols;
    if (tid >= total) return;

    uint idx = tid / ncols;
    uint col = tid % ncols;

    uint M       = 1u << (s + 1u);
    uint M_div_2 = M >> 1;
    uint M_div_4 = M >> 2;

    uint g = idx / M_div_4;
    uint q = idx % M_div_4;

    uint base = g * M + q;
    uint offA0 = base * ncols + col;
    uint offA1 = (base + M_div_4) * ncols + col;
    uint offB0 = (base + M_div_2) * ncols + col;
    uint offB1 = (base + M_div_2 + M_div_4) * ncols + col;

    ulong w_s   = twiddles[(uint)q << (stride_s1 + 1u)];
    ulong w_q   = twiddles[(uint)q << stride_s1];
    ulong w_qI  = twiddles[(uint)(q + M_div_4) << stride_s1];

    ulong x_A0 = buf[offA0];
    ulong x_A1 = buf[offA1];
    ulong x_B0 = buf[offB0];
    ulong x_B1 = buf[offB1];

    ulong t_A1 = gl_mul(x_A1, w_s);
    ulong y_A0 = gl_add(x_A0, t_A1);
    ulong y_A1 = gl_sub(x_A0, t_A1);

    ulong t_B1 = gl_mul(x_B1, w_s);
    ulong y_B0 = gl_add(x_B0, t_B1);
    ulong y_B1 = gl_sub(x_B0, t_B1);

    ulong t_B0  = gl_mul(y_B0, w_q);
    ulong t_B1p = gl_mul(y_B1, w_qI);

    buf[offA0] = gl_add(y_A0, t_B0);
    buf[offA1] = gl_add(y_A1, t_B1p);
    buf[offB0] = gl_sub(y_A0, t_B0);
    buf[offB1] = gl_sub(y_A1, t_B1p);
}

// Radix-8 phase: collapses three consecutive DIT stages (s, s+1, s+2) into
// one kernel dispatch. Each thread reads 8 elements spaced M/8 apart where
// M = 2^(s+2), runs the 3-level butterfly tree in registers, and writes the
// 8 final z-values back in place. Saves two full global-memory round-trips
// relative to plain radix-2 at the cost of 7 twiddle fetches and 12 gl_muls
// per thread.
//
// Preconditions:
//   s >= 1            (Ma_2 = 1 << (s-1) would underflow at s = 0)
//   s + 2 <= domain_pow (kernel processes stages s, s+1, s+2)
//   stride_s >= 2      (stride_b = stride_s - 1, stride_c = stride_s - 2,
//                       guarded by roots_pow - domain_pow >= 0 AND s <= domain_pow - 2,
//                       so stride_s = roots_pow - s >= 2)
//
// Twiddles: the host passes stride_s = roots_pow - s; the kernel derives
// stride_b = stride_s - 1 and stride_c = stride_s - 2 internally. This
// encodes the halving-stride structure of the roots lookup across the
// three fused stages.
kernel void ntt_radix8_phase(
    device ulong*       buf         [[buffer(0)]],
    device const ulong* twiddles    [[buffer(1)]],
    constant uint&      ncols       [[buffer(2)]],
    constant uint&      domain_size [[buffer(3)]],
    constant uint&      s           [[buffer(4)]],
    constant uint&      stride_s    [[buffer(5)]],
    uint tid                        [[thread_position_in_grid]])
{
    uint eighth = domain_size >> 3u;
    uint total  = eighth * ncols;
    if (tid >= total) return;

    uint idx = tid / ncols;
    uint col = tid % ncols;

    uint M    = 1u << (s + 2u);
    uint S8   = M >> 3u;
    uint Ma_2 = 1u << (s - 1u);

    uint g = idx / S8;
    uint q = idx % S8;

    uint base = (g * M + q) * ncols + col;
    uint step = S8 * ncols;

    ulong x0 = buf[base            ];
    ulong x1 = buf[base + 1u*step  ];
    ulong x2 = buf[base + 2u*step  ];
    ulong x3 = buf[base + 3u*step  ];
    ulong x4 = buf[base + 4u*step  ];
    ulong x5 = buf[base + 5u*step  ];
    ulong x6 = buf[base + 6u*step  ];
    ulong x7 = buf[base + 7u*step  ];

    // Stage s (innermost): 4 butterflies sharing one twiddle.
    ulong w_a = twiddles[(uint)q << stride_s];
    ulong t1 = gl_mul(x1, w_a);
    ulong t3 = gl_mul(x3, w_a);
    ulong t5 = gl_mul(x5, w_a);
    ulong t7 = gl_mul(x7, w_a);
    ulong y0 = gl_add(x0, t1);
    ulong y1 = gl_sub(x0, t1);
    ulong y2 = gl_add(x2, t3);
    ulong y3 = gl_sub(x2, t3);
    ulong y4 = gl_add(x4, t5);
    ulong y5 = gl_sub(x4, t5);
    ulong y6 = gl_add(x6, t7);
    ulong y7 = gl_sub(x6, t7);

    // Stage s+1: 2 twiddles.
    uint  stride_b = stride_s - 1u;
    ulong w_b0 = twiddles[(uint)q << stride_b];
    ulong w_b1 = twiddles[(uint)(q + Ma_2) << stride_b];
    ulong u2 = gl_mul(y2, w_b0);
    ulong u3 = gl_mul(y3, w_b1);
    ulong u6 = gl_mul(y6, w_b0);
    ulong u7 = gl_mul(y7, w_b1);
    ulong z0 = gl_add(y0, u2);
    ulong z2 = gl_sub(y0, u2);
    ulong z1 = gl_add(y1, u3);
    ulong z3 = gl_sub(y1, u3);
    ulong z4 = gl_add(y4, u6);
    ulong z6 = gl_sub(y4, u6);
    ulong z5 = gl_add(y5, u7);
    ulong z7 = gl_sub(y5, u7);

    // Stage s+2 (outermost): 4 twiddles.
    uint  stride_c = stride_s - 2u;
    ulong w_c0 = twiddles[(uint)q << stride_c];
    ulong w_c1 = twiddles[(uint)(q + Ma_2)      << stride_c];
    ulong w_c2 = twiddles[(uint)(q + 2u*Ma_2)   << stride_c];
    ulong w_c3 = twiddles[(uint)(q + 3u*Ma_2)   << stride_c];
    ulong v4 = gl_mul(z4, w_c0);
    ulong v5 = gl_mul(z5, w_c1);
    ulong v6 = gl_mul(z6, w_c2);
    ulong v7 = gl_mul(z7, w_c3);

    buf[base            ] = gl_add(z0, v4);
    buf[base + 1u*step  ] = gl_add(z1, v5);
    buf[base + 2u*step  ] = gl_add(z2, v6);
    buf[base + 3u*step  ] = gl_add(z3, v7);
    buf[base + 4u*step  ] = gl_sub(z0, v4);
    buf[base + 5u*step  ] = gl_sub(z1, v5);
    buf[base + 6u*step  ] = gl_sub(z2, v6);
    buf[base + 7u*step  ] = gl_sub(z3, v7);
}

// Fused reverse-permutation + first 3 butterfly stages, out-of-place. Each
// thread reads 8 values from `src` at their bit-reversed positions, runs
// stages s=1, s=2, s=3 in registers, then writes 8 values to `dst` at
// consecutive natural-order positions `base_nat + 0..7`.
//
// Why fused: a standalone rev-perm pass writes every byte back to global
// memory only to have the first butterfly pass read them again. Combining
// both eliminates one full domain-size read+write, which on M4 Pro is the
// dominant NTT cost for domain_pow >= 3.
//
// Stage-1 twiddles are all 1 (ω_2^0), so no gl_mul is needed there.
// Stage-2 needs ω_4 = I_val. Stage-3 needs ω_8 = W8_val and ω_8^3 = W8c_val.
// The host extracts these three constants from the roots table and passes
// them via setBytes.
//
// Requires src != dst. Not callable with domain_pow < 3 (the 8-element
// per-thread layout assumes N >= 8).
kernel void ntt_rev_butterfly_s1s2s3(
    device const ulong* src        [[buffer(0)]],
    device       ulong* dst        [[buffer(1)]],
    constant     uint&  domain_pow [[buffer(2)]],
    constant     uint&  ncols      [[buffer(3)]],
    constant     ulong& I_val      [[buffer(4)]],
    constant     ulong& W8_val     [[buffer(5)]],
    constant     ulong& W8c_val    [[buffer(6)]],
    uint tid                       [[thread_position_in_grid]])
{
    uint eighth = (1u << domain_pow) >> 3u;
    uint total  = eighth * ncols;
    if (tid >= total) return;

    uint group_idx = tid / ncols;
    uint col       = tid % ncols;

    uint shift    = 32u - domain_pow;
    uint base_nat = group_idx * 8u;

    uint ra = reverse_bits32(base_nat + 0u) >> shift;
    uint rb = reverse_bits32(base_nat + 1u) >> shift;
    uint rc = reverse_bits32(base_nat + 2u) >> shift;
    uint rd = reverse_bits32(base_nat + 3u) >> shift;
    uint re = reverse_bits32(base_nat + 4u) >> shift;
    uint rf = reverse_bits32(base_nat + 5u) >> shift;
    uint rg = reverse_bits32(base_nat + 6u) >> shift;
    uint rh = reverse_bits32(base_nat + 7u) >> shift;

    ulong x_a = src[ra * ncols + col];
    ulong x_b = src[rb * ncols + col];
    ulong x_c = src[rc * ncols + col];
    ulong x_d = src[rd * ncols + col];
    ulong x_e = src[re * ncols + col];
    ulong x_f = src[rf * ncols + col];
    ulong x_g = src[rg * ncols + col];
    ulong x_h = src[rh * ncols + col];

    // Stage s=1: 4 pair butterflies, twiddle = 1 (no gl_mul).
    ulong y0 = gl_add(x_a, x_b);
    ulong y1 = gl_sub(x_a, x_b);
    ulong y2 = gl_add(x_c, x_d);
    ulong y3 = gl_sub(x_c, x_d);
    ulong y4 = gl_add(x_e, x_f);
    ulong y5 = gl_sub(x_e, x_f);
    ulong y6 = gl_add(x_g, x_h);
    ulong y7 = gl_sub(x_g, x_h);

    // Stage s=2: (y0,y2),(y1,y3),(y4,y6),(y5,y7). Twiddles {1, I, 1, I}.
    ulong y3I = gl_mul(y3, I_val);
    ulong y7I = gl_mul(y7, I_val);
    ulong z0 = gl_add(y0, y2);
    ulong z2 = gl_sub(y0, y2);
    ulong z1 = gl_add(y1, y3I);
    ulong z3 = gl_sub(y1, y3I);
    ulong z4 = gl_add(y4, y6);
    ulong z6 = gl_sub(y4, y6);
    ulong z5 = gl_add(y5, y7I);
    ulong z7 = gl_sub(y5, y7I);

    // Stage s=3: (z0,z4),(z1,z5),(z2,z6),(z3,z7). Twiddles {1, W8, I, W8c}.
    ulong z5w = gl_mul(z5, W8_val);
    ulong z6I = gl_mul(z6, I_val);
    ulong z7c = gl_mul(z7, W8c_val);

    dst[(base_nat + 0u) * ncols + col] = gl_add(z0, z4);
    dst[(base_nat + 1u) * ncols + col] = gl_add(z1, z5w);
    dst[(base_nat + 2u) * ncols + col] = gl_add(z2, z6I);
    dst[(base_nat + 3u) * ncols + col] = gl_add(z3, z7c);
    dst[(base_nat + 4u) * ncols + col] = gl_sub(z0, z4);
    dst[(base_nat + 5u) * ncols + col] = gl_sub(z1, z5w);
    dst[(base_nat + 6u) * ncols + col] = gl_sub(z2, z6I);
    dst[(base_nat + 7u) * ncols + col] = gl_sub(z3, z7c);
}

// INTT finalisation: reorder i <-> (N - i) mod N and scale by 1/N.
//
// Math: NTT(NTT(x))[n] = N * x[(N-n) mod N], so INTT(X) = (1/N) * reorder(NTT(X)).
// By running the same forward-butterfly kernels on the frequency-domain
// input and then applying this fused reorder+scale pass, we get the inverse
// transform bit-exactly.
//
// In-place. Each thread handles one (lo=i, hi=N-i) pair across all ncols.
// For i in {0, N/2} lo == hi (fixed points of the permutation) and the
// kernel collapses to "scale in place"; the symmetric read-then-write
// pattern stays correct because both reads come from the same position.
//
// Dispatch (N/2 + 1) * ncols threads: i in [0, N/2] covers every pair exactly
// once; the `tid >= total` guard trims the non-uniform tail.
kernel void intt_reorder_scale(
    device ulong*    buf         [[buffer(0)]],
    constant uint&   domain_size [[buffer(1)]],
    constant uint&   ncols       [[buffer(2)]],
    constant ulong&  inv_n       [[buffer(3)]],
    uint tid                     [[thread_position_in_grid]])
{
    uint half_n = domain_size >> 1u;
    uint total  = (half_n + 1u) * ncols;
    if (tid >= total) return;

    uint pair_idx = tid / ncols;
    uint col      = tid % ncols;

    uint lo = pair_idx;
    uint hi = (lo == 0u) ? 0u : (domain_size - lo);

    uint lo_off = lo * ncols + col;
    uint hi_off = hi * ncols + col;

    ulong a = buf[lo_off];
    ulong b = buf[hi_off];
    buf[lo_off] = gl_canonicalize(gl_mul(b, inv_n));
    buf[hi_off] = gl_canonicalize(gl_mul(a, inv_n));
}

// Variant of intt_reorder_scale where the per-position multiplier is drawn
// from a precomputed r_inv[] array (length domain_size) instead of a single
// 1/N scalar. The host precomputes r_inv[i] = shift^i / N to combine the
// coset shift with the INTT 1/N finalisation in one pass — this is what
// pil2-stark's NTT_Goldilocks::LDE uses internally via its `extend=true`
// branch.
//
// Algebra: after forward-shaped butterflies, position i receives
// a[(N-i) mod N] in the reorder. Multiplying by r_inv[i] at write time
// gives new_buf[i] = old_buf[(N-i) mod N] * shift^i / N, which matches the
// scalar reference element-for-element.
kernel void intt_reorder_coset_scale(
    device ulong*       buf         [[buffer(0)]],
    device const ulong* r_inv       [[buffer(1)]],
    constant uint&      domain_size [[buffer(2)]],
    constant uint&      ncols       [[buffer(3)]],
    uint tid                        [[thread_position_in_grid]])
{
    uint half_n = domain_size >> 1u;
    uint total  = (half_n + 1u) * ncols;
    if (tid >= total) return;

    uint pair_idx = tid / ncols;
    uint col      = tid % ncols;

    uint lo = pair_idx;
    uint hi = (lo == 0u) ? 0u : (domain_size - lo);

    uint lo_off = lo * ncols + col;
    uint hi_off = hi * ncols + col;

    ulong a = buf[lo_off];
    ulong b = buf[hi_off];
    buf[lo_off] = gl_canonicalize(gl_mul(b, r_inv[lo]));
    buf[hi_off] = gl_canonicalize(gl_mul(a, r_inv[hi]));
}

// --- Phase D: FRI fold (cubic extension) --------------------------------
//
// Port of FRI<ElementType>::fold from pil2-stark/src/starkpil/fri/fri.hpp.
// Per-group work:
//   1. Gather nX cubic-ext values from `pol` at stride pol2N
//   2. nX-point INTT over each of 3 cubic-ext columns (base-field NTT
//      on each column independently; cubic-ext structure is preserved)
//   3. polMulAxi: scalar-mul each cubic-ext row i by sinv^i where
//      sinv = polShiftInv * wi^g for group g
//   4. Horner eval at cubic-ext `challenge`, write one cubic-ext result
//      back to pol[g]
//
// This kernel specialises to nX = 8 — the only case fibonacci-square
// (and the common case for zkEVM) hits, because each FRI step reduces
// by a factor of 8 (i.e. prevBits - currentBits == 3). A generic-nX
// variant can be added when a caller actually needs it; specialising
// lets us unroll the 3-stage butterfly and skip a roots-table lookup.
//
// Each thread handles one group g ∈ [0, pol2N). Thread-local storage:
// 24 ulong for the 8×3 cubic-ext panel (~192 bytes; within M-series
// per-thread register budget).

// Hard-coded 8-point inverse NTT on 3 independent base-field columns.
// `data[i*3 + c]` is row i, column c. After the call, data[] holds the
// bit-reversed-output of the INTT, scaled by inv(8). The twiddle table
// `roots8` has the 8-th roots of unity: roots8[k] = w^k where
// w = Goldilocks::w(3). Matches NTT_Goldilocks(8).INTT(dst, src, 8, 3)
// exactly for the data-flow convention used by FRI::fold.
//
// Algorithm mirrors what NTT_Goldilocks::NTT does for inverse=true:
//   1. bit-reverse permutation (for N=8: 0<->0, 1<->4, 2<->2, 3<->6,
//      4<->1, 5<->5, 6<->3, 7<->7 — swap pairs where rev(i) > i: (1,4)
//      and (3,6)).
//   2. 3 stages of radix-2 butterflies, twiddles from roots8
//   3. multiply by inv(8) scalar
// One radix-2 cubic-ext butterfly. (a, b) → (a + w*b, a - w*b).
inline void fri_bf_ext3(thread ulong (&d)[24],
                        uint a_row,
                        uint b_row,
                        ulong w) {
    thread ulong ra[3] = { d[a_row*3+0], d[a_row*3+1], d[a_row*3+2] };
    thread ulong rb[3] = { d[b_row*3+0], d[b_row*3+1], d[b_row*3+2] };
    thread ulong wb[3];
    gl3_mul_scalar(wb, rb, w);
    thread ulong lh[3];
    thread ulong hh[3];
    gl3_add(lh, ra, wb);
    gl3_sub(hh, ra, wb);
    d[a_row*3+0] = lh[0]; d[a_row*3+1] = lh[1]; d[a_row*3+2] = lh[2];
    d[b_row*3+0] = hh[0]; d[b_row*3+1] = hh[1]; d[b_row*3+2] = hh[2];
}

inline void fri_intt8_ext3(thread ulong (&data)[24],
                           device const ulong* roots8,
                           ulong inv8) {
    // Bit-reverse: N=8, rev(1)=4, rev(3)=6.
    {
        // swap row 1 <-> row 4
        ulong t0 = data[1*3+0]; data[1*3+0] = data[4*3+0]; data[4*3+0] = t0;
        ulong t1 = data[1*3+1]; data[1*3+1] = data[4*3+1]; data[4*3+1] = t1;
        ulong t2 = data[1*3+2]; data[1*3+2] = data[4*3+2]; data[4*3+2] = t2;
        // swap row 3 <-> row 6
        ulong u0 = data[3*3+0]; data[3*3+0] = data[6*3+0]; data[6*3+0] = u0;
        ulong u1 = data[3*3+1]; data[3*3+1] = data[6*3+1]; data[6*3+1] = u1;
        ulong u2 = data[3*3+2]; data[3*3+2] = data[6*3+2]; data[6*3+2] = u2;
    }

    // Twiddle lookups: for inverse NTT over size 8 with our roots
    // convention (roots[k] = w^k, w = 8-th root), the inverse uses
    // w^(-k) = w^(N-k) = roots[N-k]. NTT_Goldilocks.INTT achieves this
    // internally via the intt_idx(i, N) helper (i==0 ? 0 : N - i).
    // Stage s (1-indexed) processes butterflies of half-stride m = 2^s.
    // Twiddle at butterfly index j within the stage: roots8[(N-j) mod N]
    // with stride = (roots_len / m) — but we use roots_len = N = 8, so
    // the stride is 1 and indexing simplifies.
    //
    // Butterfly (a, b) with twiddle w: a' = a + w*b; b' = a - w*b.

    // For each stage, iterate over butterfly groups. For N=8:
    //   stage 1: m=2, 4 butterflies of stride 1: (0,1), (2,3), (4,5), (6,7)
    //     all use twiddle = 1 (w^0)
    //   stage 2: m=4, 4 butterflies of stride 2: (0,2), (1,3), (4,6), (5,7)
    //     twiddles: 1, w^(-2) = roots8[6]   (for inverse)
    //   stage 3: m=8, 4 butterflies of stride 4: (0,4), (1,5), (2,6), (3,7)
    //     twiddles: 1, w^(-1), w^(-2), w^(-3) = roots8[0], [7], [6], [5]
    //
    // Encoded below; each butterfly works across the 3 cubic-ext columns
    // via the scalar-cubic scaling gl3_mul_scalar and gl3_add/gl3_sub.

    // Stage 1: 4 butterflies, twiddle = 1 (roots8[0]).
    fri_bf_ext3(data, 0, 1, roots8[0]);
    fri_bf_ext3(data, 2, 3, roots8[0]);
    fri_bf_ext3(data, 4, 5, roots8[0]);
    fri_bf_ext3(data, 6, 7, roots8[0]);

    // Stage 2: 4 butterflies, twiddles = roots8[0] and roots8[6]
    //          (inverse uses w^(-j) = roots8[N - j], so j=0→[0], j=1→[6]
    //           for the stride-of-2 inverse stage where the forward had
    //           j indexing 0..1 with twiddle = w^j at m=4, i.e. w^{N/m * j}
    //           = w^{2j}). Our roots table is the FORWARD primitive; the
    //           inverse-NTT convention in NTT_Goldilocks flips via the
    //           intt_idx helper.
    //
    // Twiddle for butterfly index j at stage s in forward NTT of size N:
    //   w^{(N / 2^s) * j}
    // with s=2, N=8: w^(2*j), j ∈ {0, 1}. Forward twiddles: w^0, w^2.
    // Inverse twiddles: w^0, w^{-2} = w^{8-2} = w^6 = roots8[6].
    fri_bf_ext3(data, 0, 2, roots8[0]);
    fri_bf_ext3(data, 1, 3, roots8[6]);
    fri_bf_ext3(data, 4, 6, roots8[0]);
    fri_bf_ext3(data, 5, 7, roots8[6]);

    // Stage 3: 4 butterflies, stride = 4. Forward twiddles: w^j for j=0..3.
    // Inverse: w^{-j} = roots8[(8 - j) mod 8].
    fri_bf_ext3(data, 0, 4, roots8[0]);
    fri_bf_ext3(data, 1, 5, roots8[7]);
    fri_bf_ext3(data, 2, 6, roots8[6]);
    fri_bf_ext3(data, 3, 7, roots8[5]);

    // Scale by inv(8).
    for (uint i = 0; i < 24; i++) {
        data[i] = gl_canonicalize(gl_mul(data[i], inv8));
    }
}

kernel void fri_fold_w8_k(
    device       ulong* pol           [[buffer(0)]],
    device const ulong* challenge     [[buffer(1)]],  // 3 u64
    device const ulong* roots8        [[buffer(2)]],  // 8 u64
    constant uint&      pol2N         [[buffer(3)]],
    constant ulong&     polShiftInv   [[buffer(4)]],
    constant ulong&     wi            [[buffer(5)]],
    constant ulong&     inv8          [[buffer(6)]],
    uint tid                          [[thread_position_in_grid]])
{
    if (tid >= pol2N) return;
    const uint g = tid;
    const uint nX = 8u;

    // Step 1: compute sinv = polShiftInv * wi^g via square-and-multiply.
    // The intermediates only feed further gl_muls; gl_mul itself is lazy-
    // safe on its inputs, so we can drop the per-step canonicalise. Final
    // sinv left lazy too -- downstream consumers (gl3_mul_scalar inside
    // Step 4) all go through gl_mul which accepts any u64.
    ulong wi_pow_g = 1UL;
    ulong base     = wi;
    uint  e        = g;
    while (e > 0u) {
        if ((e & 1u) != 0u) wi_pow_g = gl_mul(wi_pow_g, base);
        base = gl_mul(base, base);
        e >>= 1u;
    }
    const ulong sinv = gl_mul(polShiftInv, wi_pow_g);

    // Step 2: gather nX cubic-ext values at stride pol2N.
    thread ulong ppar[24];
    for (uint i = 0u; i < nX; i++) {
        const uint src = ((i * pol2N) + g) * 3u;
        ppar[i*3+0] = pol[src+0];
        ppar[i*3+1] = pol[src+1];
        ppar[i*3+2] = pol[src+2];
    }

    // Step 3: 8-point INTT on each cubic-ext column, in-place.
    fri_intt8_ext3(ppar, roots8, inv8);

    // Step 4: polMulAxi — multiply row i by sinv^i (scalar × cubic-ext).
    ulong r = 1UL;
    for (uint i = 0u; i < nX; i++) {
        thread ulong row[3] = { ppar[i*3+0], ppar[i*3+1], ppar[i*3+2] };
        thread ulong scaled[3];
        gl3_mul_scalar(scaled, row, r);
        ppar[i*3+0] = scaled[0];
        ppar[i*3+1] = scaled[1];
        ppar[i*3+2] = scaled[2];
        r = gl_mul(r, sinv);
    }

    // Step 5: Horner evaluation at `challenge` (cubic-ext), top-down.
    thread ulong chal[3] = { challenge[0], challenge[1], challenge[2] };
    thread ulong acc[3]  = { ppar[(nX-1)*3+0], ppar[(nX-1)*3+1], ppar[(nX-1)*3+2] };
    for (int i = int(nX) - 2; i >= 0; i--) {
        thread ulong aux[3];
        gl3_mul(aux, acc, chal);
        thread ulong pi[3] = { ppar[i*3+0], ppar[i*3+1], ppar[i*3+2] };
        gl3_add(acc, aux, pi);
    }

    pol[g*3+0] = acc[0];
    pol[g*3+1] = acc[1];
    pol[g*3+2] = acc[2];
}

// --- Phase C7: Poseidon2 permutation (W=8) ------------------------------
//
// Port of Poseidon2Goldilocks<8>::permute_seq from pil2-stark. Same math:
//   matmul_external(state)
//   for r=0..3: add_constants + pow7 per element; matmul_external
//   for r=0..21: add constant to state[0]; pow7 state[0]; state[i] = state[i]*D[i] + sum(state)
//   for r=0..3: add_constants + pow7; matmul_external
//
// Constants C (length 86 = 32 + 22 + 32) and D (length 8) are passed as
// device buffers by the host — the .mm bridge pulls them from
// Poseidon2GoldilocksConstants::C8 / D8 so we don't need to duplicate
// the tables in MSL source.
//
// Each thread processes one independent sponge. Batched hashing dispatches
// `count` threads; the per-thread state sits in registers (8 ulong = 64 B,
// well within the M-series per-thread register budget).

inline void pose2_matmul_m4(thread ulong* s) {
    ulong t0   = gl_add(s[0], s[1]);
    ulong t1   = gl_add(s[2], s[3]);
    ulong t2   = gl_add(gl_add(s[1], s[1]), t1);
    ulong t3   = gl_add(gl_add(s[3], s[3]), t0);
    ulong t0_2 = gl_add(t0, t0);
    ulong t1_2 = gl_add(t1, t1);
    ulong t4   = gl_add(gl_add(t1_2, t1_2), t3);
    ulong t5   = gl_add(gl_add(t0_2, t0_2), t2);
    ulong t6   = gl_add(t3, t5);
    ulong t7   = gl_add(t2, t4);
    s[0] = t6; s[1] = t5; s[2] = t7; s[3] = t4;
}

// W-parameterised matmul_external: apply matmul_m4 to each 4-block then
// accumulate the column sums across blocks and fold back. Works for W ∈
// {8, 12, 16}; the MSL compiler unrolls since W is a constant at the call
// site.
inline void pose2_matmul_external_n(thread ulong* s, const uint W) {
    for (uint i = 0; i < W; i += 4) {
        pose2_matmul_m4(&s[i]);
    }
    // W-specialised pairwise reduction. For W=16 this drops the accumulator
    // chain depth from 4 sequential adds per slot to 2 (pair + pair-of-pairs),
    // widening the window the compiler can schedule against the four
    // independent per-column chains. For W=12 depth drops 3→2; W=8 is
    // already depth 1 so the linear chain is kept.
    ulong stored[4];
    if (W == 16u) {
        ulong p00 = gl_add(s[0],  s[4]);
        ulong p01 = gl_add(s[8],  s[12]);
        ulong p10 = gl_add(s[1],  s[5]);
        ulong p11 = gl_add(s[9],  s[13]);
        ulong p20 = gl_add(s[2],  s[6]);
        ulong p21 = gl_add(s[10], s[14]);
        ulong p30 = gl_add(s[3],  s[7]);
        ulong p31 = gl_add(s[11], s[15]);
        stored[0] = gl_add(p00, p01);
        stored[1] = gl_add(p10, p11);
        stored[2] = gl_add(p20, p21);
        stored[3] = gl_add(p30, p31);
    } else if (W == 12u) {
        stored[0] = gl_add(gl_add(s[0], s[4]), s[8]);
        stored[1] = gl_add(gl_add(s[1], s[5]), s[9]);
        stored[2] = gl_add(gl_add(s[2], s[6]), s[10]);
        stored[3] = gl_add(gl_add(s[3], s[7]), s[11]);
    } else {  // W == 8
        stored[0] = gl_add(s[0], s[4]);
        stored[1] = gl_add(s[1], s[5]);
        stored[2] = gl_add(s[2], s[6]);
        stored[3] = gl_add(s[3], s[7]);
    }
    for (uint i = 0; i < W; i++) {
        s[i] = gl_add(s[i], stored[i & 3u]);
    }
}

inline ulong pose2_pow7(ulong x) {
    // gl_mul's output is "lazy" (u64 ≡ result mod p, but not canonical);
    // gl_add/gl_sub both ALSO tolerate any u64 input via their
    // carry/borrow bookkeeping. Every consumer of a pow7 result is one
    // of those -- matmul's gl_add chains, pow7add_n's gl_add, partial
    // rounds' gl_mul -- so nothing downstream needs the canonical form.
    // Drop the final canonicalise too; the kernel-level canonicalise on
    // the final digest store (in the leaf/compress/linear_hash kernels)
    // is what matters for bit-exactness against the CPU reference.
    ulong x2 = gl_mul(x, x);
    ulong x3 = gl_mul(x, x2);
    ulong x4 = gl_mul(x2, x2);
    return gl_mul(x3, x4);
}

inline void pose2_pow7add_n(thread ulong* s,
                             device const ulong* C,
                             const uint W) {
    for (uint i = 0; i < W; i++) {
        ulong xi = gl_add(s[i], C[i]);
        s[i] = pose2_pow7(xi);
    }
}

// --- Threadgroup-memory state variants ----------------------------------
//
// The thread-local `ulong s[W]` form above uses 8W bytes of per-thread
// registers; with W=16 the compiler pins ~16 u64 registers plus
// temporaries, landing around 50-60 32-bit GPRs per thread. GPU profile
// data (captured via MTLCaptureManager) showed this caps Kernel Occupancy
// at 28% (Occupancy Manager Target 29.97%) with L1 Register Residency
// 53% and Stack L1 Write Bandwidth 134 GB/s -- i.e. register spills into
// stack memory.
//
// Moving state to threadgroup memory lets the register allocator see
// only the CURRENT m4 block (4 u64) or the current accumulator (1-4 u64)
// as live, so per-thread register count drops dramatically. The
// tradeoff is one load + one store to threadgroup memory per state
// access; on Apple Silicon that's ~1-2 cycles vs 0 for a register,
// but higher occupancy more than pays it back when the kernel is
// register-bound, as the merkle workload is.
//
// Per-thread state slice is `tg_state[local_tid * W .. local_tid * W + W)`.
// No synchronisation needed: each thread only reads/writes its own slice.

inline void pose2_matmul_m4_tg(threadgroup ulong* s) {
    // Copy the 4-block into registers, run the m4 math entirely in
    // registers, then write back. Peak register count during this
    // function: 4 loaded + ~4 temps = ~8 u64.
    //
    // The m4 matmul is [5 7 1 3; 4 6 1 1; 1 3 5 7; 1 1 4 6]. The
    // 4*u terms are done via gl_mul_small(u, 4) instead of a chain
    // of two gl_adds (y=u+u, z=y+y). This trims 2 gl_add calls (each
    // ~10 integer-conditional ops) down to one gl_mul_small (~8
    // integer-complex ops) per 4*u computation. Two such terms per
    // m4 call (the 4*t0 and 4*t1 for s[1], s[3] outputs) = ~24 ops
    // moved off the Conditional pipeline (at 83% limiter) onto the
    // Complex pipeline (at 62% limiter, headroom).
    ulong a0 = s[0];
    ulong a1 = s[1];
    ulong a2 = s[2];
    ulong a3 = s[3];
    ulong t0   = gl_add(a0, a1);
    ulong t1   = gl_add(a2, a3);
    ulong t2   = gl_add(gl_add(a1, a1), t1);
    ulong t3   = gl_add(gl_add(a3, a3), t0);
    ulong t0_4 = gl_mul_small(t0, 4u);
    ulong t1_4 = gl_mul_small(t1, 4u);
    ulong t4   = gl_add(t1_4, t3);
    ulong t5   = gl_add(t0_4, t2);
    ulong t6   = gl_add(t3, t5);
    ulong t7   = gl_add(t2, t4);
    s[0] = t6; s[1] = t5; s[2] = t7; s[3] = t4;
}

inline void pose2_matmul_external_n_tg(threadgroup ulong* s, const uint W) {
    for (uint i = 0; i < W; i += 4) {
        pose2_matmul_m4_tg(&s[i]);
    }
    // Pairwise column-sum reduction, W-specialised. Same shape as the
    // thread-state version (see commit 6901b051) but reads through
    // threadgroup memory.
    ulong stored0, stored1, stored2, stored3;
    if (W == 16u) {
        ulong p00 = gl_add(s[0],  s[4]);
        ulong p01 = gl_add(s[8],  s[12]);
        ulong p10 = gl_add(s[1],  s[5]);
        ulong p11 = gl_add(s[9],  s[13]);
        ulong p20 = gl_add(s[2],  s[6]);
        ulong p21 = gl_add(s[10], s[14]);
        ulong p30 = gl_add(s[3],  s[7]);
        ulong p31 = gl_add(s[11], s[15]);
        stored0 = gl_add(p00, p01);
        stored1 = gl_add(p10, p11);
        stored2 = gl_add(p20, p21);
        stored3 = gl_add(p30, p31);
    } else if (W == 12u) {
        stored0 = gl_add(gl_add(s[0], s[4]), s[8]);
        stored1 = gl_add(gl_add(s[1], s[5]), s[9]);
        stored2 = gl_add(gl_add(s[2], s[6]), s[10]);
        stored3 = gl_add(gl_add(s[3], s[7]), s[11]);
    } else {  // W == 8
        stored0 = gl_add(s[0], s[4]);
        stored1 = gl_add(s[1], s[5]);
        stored2 = gl_add(s[2], s[6]);
        stored3 = gl_add(s[3], s[7]);
    }
    // Fold back into each block. Keep the 4 stored values in registers
    // for the duration of this loop.
    for (uint i = 0; i < W; i += 4) {
        s[i + 0] = gl_add(s[i + 0], stored0);
        s[i + 1] = gl_add(s[i + 1], stored1);
        s[i + 2] = gl_add(s[i + 2], stored2);
        s[i + 3] = gl_add(s[i + 3], stored3);
    }
}

inline void pose2_pow7add_n_tg(threadgroup ulong* s,
                                device const ulong* C,
                                const uint W) {
    for (uint i = 0; i < W; i++) {
        ulong xi = gl_add(s[i], C[i]);
        s[i] = pose2_pow7(xi);
    }
}

inline void pose2_permute_n_inplace_tg(threadgroup ulong* s,
                                        device const ulong* C,
                                        device const ulong* D,
                                        const uint W)
{
    const uint RF_HALF = 4u;
    const uint RP      = 22u;

    pose2_matmul_external_n_tg(s, W);

    for (uint r = 0; r < RF_HALF; r++) {
        pose2_pow7add_n_tg(s, C + r * W, W);
        pose2_matmul_external_n_tg(s, W);
    }

    for (uint r = 0; r < RP; r++) {
        // Partial round: read s[0], transform, then sum all W elements.
        // Keep sum in a register across both sub-loops.
        ulong s0 = gl_add(s[0], C[RF_HALF * W + r]);
        s0 = pose2_pow7(s0);
        s[0] = s0;
        ulong sum = s0;
        for (uint i = 1; i < W; i++) sum = gl_add(sum, s[i]);
        for (uint i = 0; i < W; i++) {
            s[i] = gl_add(gl_mul(s[i], D[i]), sum);
        }
    }

    const uint final_base = RF_HALF * W + RP;
    for (uint r = 0; r < RF_HALF; r++) {
        pose2_pow7add_n_tg(s, C + final_base + r * W, W);
        pose2_matmul_external_n_tg(s, W);
    }
}

// (2x-interleaved permute experiment removed. Tried two permutes per
// thread with ops interleaved across s0/s1 for scheduler ILP. Doubling
// TG memory per-TG halved concurrent TGs per SM; the ~44% ALU
// utilisation baseline didn't translate into a 2x win. Measured ~60%
// regression on fibonacci-square merkle gpu_only vs single-permute.
// GPU's thread-level parallelism was already saturating issue width.)

// W-parameterised full Poseidon2 permutation. Used by the per-W kernels
// below. RP=22 holds for W ∈ {8, 12, 16}; W=4 would need RP=21 and is not
// implemented here.
inline void pose2_permute_n_inplace(thread ulong* s,
                                    device const ulong* C,
                                    device const ulong* D,
                                    const uint W)
{
    const uint RF_HALF = 4u;
    const uint RP      = 22u;

    pose2_matmul_external_n(s, W);

    for (uint r = 0; r < RF_HALF; r++) {
        pose2_pow7add_n(s, C + r * W, W);
        pose2_matmul_external_n(s, W);
    }

    for (uint r = 0; r < RP; r++) {
        s[0] = gl_add(s[0], C[RF_HALF * W + r]);
        s[0] = pose2_pow7(s[0]);
        ulong sum = s[0];
        for (uint i = 1; i < W; i++) sum = gl_add(sum, s[i]);
        // gl_add handles any u64 inputs correctly (its carry bookkeeping
        // folds 2^64 ≡ 2^32-1 mod p), and its output is always canonical
        // via the final (s >= p) subtract. So we can drop the canonicalize
        // that used to sit between gl_mul and gl_add — gl_mul's lazy u64
        // output feeds straight in. 16×22 = 352 saved per W=16 permute.
        for (uint i = 0; i < W; i++) {
            s[i] = gl_add(gl_mul(s[i], D[i]), sum);
        }
    }

    const uint final_base = RF_HALF * W + RP;
    for (uint r = 0; r < RF_HALF; r++) {
        pose2_pow7add_n(s, C + final_base + r * W, W);
        pose2_matmul_external_n(s, W);
    }
}

// Per-thread state now lives in threadgroup memory (see rationale above
// the pose2_*_tg helpers). Threadgroup size is fixed at 64 so the
// dispatcher can rely on the TG-memory layout; the dispatch site still
// asks the PSO for its max but caps at 64, which is the common case.
constant constexpr uint kPose2TGSize = 64u;

kernel void poseidon2_permute_w8_batch(
    device const ulong* in_states  [[buffer(0)]],
    device       ulong* out_states [[buffer(1)]],
    device const ulong* C          [[buffer(2)]],  // length 86
    device const ulong* D          [[buffer(3)]],  // length 8
    constant uint&      count      [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]])
{
    threadgroup ulong tg_state[kPose2TGSize * 8u];
    threadgroup ulong* s = &tg_state[lid * 8u];
    if (tid >= count) return;
    for (uint i = 0; i < 8u; i++) s[i] = in_states[tid * 8u + i];
    pose2_permute_n_inplace_tg(s, C, D, 8u);
    for (uint i = 0; i < 8u; i++) out_states[tid * 8u + i] = gl_canonicalize(s[i]);
}

kernel void poseidon2_permute_w12_batch(
    device const ulong* in_states  [[buffer(0)]],
    device       ulong* out_states [[buffer(1)]],
    device const ulong* C          [[buffer(2)]],  // length 118
    device const ulong* D          [[buffer(3)]],  // length 12
    constant uint&      count      [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]])
{
    threadgroup ulong tg_state[kPose2TGSize * 12u];
    threadgroup ulong* s = &tg_state[lid * 12u];
    if (tid >= count) return;
    for (uint i = 0; i < 12u; i++) s[i] = in_states[tid * 12u + i];
    pose2_permute_n_inplace_tg(s, C, D, 12u);
    for (uint i = 0; i < 12u; i++) out_states[tid * 12u + i] = gl_canonicalize(s[i]);
}

kernel void poseidon2_permute_w16_batch(
    device const ulong* in_states  [[buffer(0)]],
    device       ulong* out_states [[buffer(1)]],
    device const ulong* C          [[buffer(2)]],  // length 150
    device const ulong* D          [[buffer(3)]],  // length 16
    constant uint&      count      [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]])
{
    threadgroup ulong tg_state[kPose2TGSize * 16u];
    threadgroup ulong* s = &tg_state[lid * 16u];
    if (tid >= count) return;
    for (uint i = 0; i < 16u; i++) s[i] = in_states[tid * 16u + i];
    pose2_permute_n_inplace_tg(s, C, D, 16u);
    for (uint i = 0; i < 16u; i++) out_states[tid * 16u + i] = gl_canonicalize(s[i]);
}

// Merkle leaf hash for W=8, RATE=4, CAPACITY=4. Each thread reads RATE=4
// u64s from `in_rate`, constructs the state [in_rate, 0, 0, 0, 0] to match
// linear_hash_seq's first-iteration layout, permutes, and writes the first
// CAPACITY=4 result elements to `out_digest`.
//
// Caller contract: input is stored as `count * RATE` consecutive u64s;
// output is `count * CAPACITY` consecutive u64s.
kernel void pose2_leaf_hash_w8(
    device const ulong* in_rate    [[buffer(0)]],
    device       ulong* out_digest [[buffer(1)]],
    device const ulong* C          [[buffer(2)]],
    device const ulong* D          [[buffer(3)]],
    constant uint&      count      [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]])
{
    threadgroup ulong tg_state[kPose2TGSize * 8u];
    threadgroup ulong* s = &tg_state[lid * 8u];
    if (tid >= count) return;

    s[0] = in_rate[tid * 4u + 0u];
    s[1] = in_rate[tid * 4u + 1u];
    s[2] = in_rate[tid * 4u + 2u];
    s[3] = in_rate[tid * 4u + 3u];
    s[4] = 0UL;
    s[5] = 0UL;
    s[6] = 0UL;
    s[7] = 0UL;

    pose2_permute_n_inplace_tg(s, C, D, 8u);

    out_digest[tid * 4u + 0u] = gl_canonicalize(s[0]);
    out_digest[tid * 4u + 1u] = gl_canonicalize(s[1]);
    out_digest[tid * 4u + 2u] = gl_canonicalize(s[2]);
    out_digest[tid * 4u + 3u] = gl_canonicalize(s[3]);
}

// Merkle parent compress for W=8. Each thread reads W=8 u64s (two child
// CAPACITY=4 digests concatenated), permutes, writes the first CAPACITY=4
// result elements as the parent digest. Matches the scalar compress_seq.
//
// Host invokes this per Merkle layer with different buffer offsets to walk
// up the tree in-place.
kernel void pose2_compress_w8(
    device const ulong* in_pair    [[buffer(0)]],
    device       ulong* out_parent [[buffer(1)]],
    device const ulong* C          [[buffer(2)]],
    device const ulong* D          [[buffer(3)]],
    constant uint&      count      [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]])
{
    threadgroup ulong tg_state[kPose2TGSize * 8u];
    threadgroup ulong* s = &tg_state[lid * 8u];
    if (tid >= count) return;

    for (uint i = 0; i < 8; i++) s[i] = in_pair[tid * 8u + i];

    pose2_permute_n_inplace_tg(s, C, D, 8u);

    out_parent[tid * 4u + 0u] = gl_canonicalize(s[0]);
    out_parent[tid * 4u + 1u] = gl_canonicalize(s[1]);
    out_parent[tid * 4u + 2u] = gl_canonicalize(s[2]);
    out_parent[tid * 4u + 3u] = gl_canonicalize(s[3]);
}

// W=12 leaf hash (RATE=8, CAPACITY=4). Matches linear_hash_seq for
// size == RATE: state = [input[0..8], 0, 0, 0, 0], permute, output first 4.
kernel void pose2_leaf_hash_w12(
    device const ulong* in_rate    [[buffer(0)]],   // count * 8
    device       ulong* out_digest [[buffer(1)]],   // count * 4
    device const ulong* C          [[buffer(2)]],
    device const ulong* D          [[buffer(3)]],
    constant uint&      count      [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]])
{
    threadgroup ulong tg_state[kPose2TGSize * 12u];
    threadgroup ulong* s = &tg_state[lid * 12u];
    if (tid >= count) return;
    for (uint i = 0; i < 8u; i++) s[i] = in_rate[tid * 8u + i];
    s[8] = 0UL; s[9] = 0UL; s[10] = 0UL; s[11] = 0UL;
    pose2_permute_n_inplace_tg(s, C, D, 12u);
    for (uint i = 0; i < 4u; i++) out_digest[tid * 4u + i] = gl_canonicalize(s[i]);
}

// W=12 arity-3 compress: 3 children × CAPACITY=4 = 12 u64s fills the state
// exactly (no zero padding). Permute, output first CAPACITY.
kernel void pose2_compress_w12(
    device const ulong* in_triple  [[buffer(0)]],   // count * 12
    device       ulong* out_parent [[buffer(1)]],   // count * 4
    device const ulong* C          [[buffer(2)]],
    device const ulong* D          [[buffer(3)]],
    constant uint&      count      [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]])
{
    threadgroup ulong tg_state[kPose2TGSize * 12u];
    threadgroup ulong* s = &tg_state[lid * 12u];
    if (tid >= count) return;
    for (uint i = 0; i < 12u; i++) s[i] = in_triple[tid * 12u + i];
    pose2_permute_n_inplace_tg(s, C, D, 12u);
    for (uint i = 0; i < 4u; i++) out_parent[tid * 4u + i] = gl_canonicalize(s[i]);
}

// W=16 leaf hash (RATE=12, CAPACITY=4).
kernel void pose2_leaf_hash_w16(
    device const ulong* in_rate    [[buffer(0)]],   // count * 12
    device       ulong* out_digest [[buffer(1)]],   // count * 4
    device const ulong* C          [[buffer(2)]],
    device const ulong* D          [[buffer(3)]],
    constant uint&      count      [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]])
{
    threadgroup ulong tg_state[kPose2TGSize * 16u];
    threadgroup ulong* s = &tg_state[lid * 16u];
    if (tid >= count) return;
    for (uint i = 0; i < 12u; i++) s[i] = in_rate[tid * 12u + i];
    s[12] = 0UL; s[13] = 0UL; s[14] = 0UL; s[15] = 0UL;
    pose2_permute_n_inplace_tg(s, C, D, 16u);
    for (uint i = 0; i < 4u; i++) out_digest[tid * 4u + i] = gl_canonicalize(s[i]);
}

// W=16 arity-4 compress: 4 children × CAPACITY=4 = 16 u64s fills the state
// exactly. Permute, output first CAPACITY.
kernel void pose2_compress_w16(
    device const ulong* in_quad    [[buffer(0)]],   // count * 16
    device       ulong* out_parent [[buffer(1)]],   // count * 4
    device const ulong* C          [[buffer(2)]],
    device const ulong* D          [[buffer(3)]],
    constant uint&      count      [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]])
{
    threadgroup ulong tg_state[kPose2TGSize * 16u];
    threadgroup ulong* s = &tg_state[lid * 16u];
    if (tid >= count) return;
    for (uint i = 0; i < 16u; i++) s[i] = in_quad[tid * 16u + i];
    pose2_permute_n_inplace_tg(s, C, D, 16u);
    for (uint i = 0; i < 4u; i++) out_parent[tid * 4u + i] = gl_canonicalize(s[i]);
}

// Sponge-absorb leaf hash for W=16: bit-exact with
// Poseidon2Goldilocks<16>::linear_hash_seq when RATE=12, CAPACITY=4.
// Each thread reads a `num_cols` row from `input` and produces CAPACITY
// u64s in `out_digest`. State absorbs up to RATE elements per iteration;
// on the first iteration capacity is zero, on every subsequent iteration
// capacity is the previous permute's state[0..CAPACITY]. Final
// state[0..CAPACITY] is the digest.
//
// num_cols == 0 short-circuits to all zeros (linear_hash_seq's else
// branch). Any non-zero num_cols enters the loop at least once, even if
// num_cols < CAPACITY — linear_hash_seq runs at least one permute.
kernel void pose2_linear_hash_w16(
    device const ulong* input      [[buffer(0)]],   // count * num_cols
    device       ulong* out_digest [[buffer(1)]],   // count * 4
    device const ulong* C          [[buffer(2)]],
    device const ulong* D          [[buffer(3)]],
    constant uint&      count      [[buffer(4)]],
    constant uint&      num_cols   [[buffer(5)]],
    uint tid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]])
{
    threadgroup ulong tg_state[kPose2TGSize * 16u];
    threadgroup ulong* state = &tg_state[lid * 16u];
    if (tid >= count) return;

    const uint RATE     = 12u;
    const uint CAPACITY = 4u;
    const uint W        = 16u;

    if (num_cols == 0u) {
        for (uint i = 0; i < CAPACITY; i++) out_digest[tid * CAPACITY + i] = 0UL;
        return;
    }

    const device ulong* in_row = input + (ulong)tid * (ulong)num_cols;

    // First absorb: capacity = 0.
    {
        uint n = (num_cols < RATE) ? num_cols : RATE;
        for (uint i = 0; i < n; i++) state[i] = in_row[i];
        for (uint i = n; i < RATE; i++) state[i] = 0UL;
        for (uint i = 0; i < CAPACITY; i++) state[RATE + i] = 0UL;
        pose2_permute_n_inplace_tg(state, C, D, W);
    }

    uint offset    = (num_cols < RATE) ? num_cols : RATE;
    uint remaining = num_cols - offset;

    while (remaining > 0u) {
        // Rotate previous digest into capacity slots. RATE (12) >= CAPACITY
        // (4), so state[0..CAPACITY] never overlaps state[RATE..RATE+CAPACITY].
        for (uint i = 0; i < CAPACITY; i++) state[RATE + i] = state[i];
        uint n = (remaining < RATE) ? remaining : RATE;
        for (uint i = 0; i < n; i++)      state[i] = in_row[offset + i];
        for (uint i = n; i < RATE; i++)   state[i] = 0UL;
        pose2_permute_n_inplace_tg(state, C, D, W);
        offset    += n;
        remaining -= n;
    }

    for (uint i = 0; i < CAPACITY; i++) {
        out_digest[tid * CAPACITY + i] = gl_canonicalize(state[i]);
    }
}

// Sponge-absorb leaf hash for W=12 (arity-3 Merkle). Identical algorithm
// to pose2_linear_hash_w16 but with RATE=8, CAPACITY=4, W=12. Bit-exact
// with Poseidon2Goldilocks<12>::linear_hash_seq.
kernel void pose2_linear_hash_w12(
    device const ulong* input      [[buffer(0)]],
    device       ulong* out_digest [[buffer(1)]],
    device const ulong* C          [[buffer(2)]],
    device const ulong* D          [[buffer(3)]],
    constant uint&      count      [[buffer(4)]],
    constant uint&      num_cols   [[buffer(5)]],
    uint tid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]])
{
    threadgroup ulong tg_state[kPose2TGSize * 12u];
    threadgroup ulong* state = &tg_state[lid * 12u];
    if (tid >= count) return;

    const uint RATE     = 8u;
    const uint CAPACITY = 4u;
    const uint W        = 12u;

    if (num_cols == 0u) {
        for (uint i = 0; i < CAPACITY; i++) out_digest[tid * CAPACITY + i] = 0UL;
        return;
    }

    const device ulong* in_row = input + (ulong)tid * (ulong)num_cols;

    {
        uint n = (num_cols < RATE) ? num_cols : RATE;
        for (uint i = 0; i < n; i++) state[i] = in_row[i];
        for (uint i = n; i < RATE; i++) state[i] = 0UL;
        for (uint i = 0; i < CAPACITY; i++) state[RATE + i] = 0UL;
        pose2_permute_n_inplace_tg(state, C, D, W);
    }

    uint offset    = (num_cols < RATE) ? num_cols : RATE;
    uint remaining = num_cols - offset;

    while (remaining > 0u) {
        // RATE (8) >= CAPACITY (4), so state[0..CAPACITY] doesn't overlap
        // state[RATE..RATE+CAPACITY].
        for (uint i = 0; i < CAPACITY; i++) state[RATE + i] = state[i];
        uint n = (remaining < RATE) ? remaining : RATE;
        for (uint i = 0; i < n; i++)      state[i] = in_row[offset + i];
        for (uint i = n; i < RATE; i++)   state[i] = 0UL;
        pose2_permute_n_inplace_tg(state, C, D, W);
        offset    += n;
        remaining -= n;
    }

    for (uint i = 0; i < CAPACITY; i++) {
        out_digest[tid * CAPACITY + i] = gl_canonicalize(state[i]);
    }
}

// Sponge-absorb leaf hash for W=8 (arity-2 Merkle). RATE=4, CAPACITY=4.
// Bit-exact with Poseidon2Goldilocks<8>::linear_hash_seq.
kernel void pose2_linear_hash_w8(
    device const ulong* input      [[buffer(0)]],
    device       ulong* out_digest [[buffer(1)]],
    device const ulong* C          [[buffer(2)]],
    device const ulong* D          [[buffer(3)]],
    constant uint&      count      [[buffer(4)]],
    constant uint&      num_cols   [[buffer(5)]],
    uint tid                       [[thread_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]])
{
    threadgroup ulong tg_state[kPose2TGSize * 8u];
    threadgroup ulong* state = &tg_state[lid * 8u];
    if (tid >= count) return;

    const uint RATE     = 4u;
    const uint CAPACITY = 4u;
    const uint W        = 8u;

    if (num_cols == 0u) {
        for (uint i = 0; i < CAPACITY; i++) out_digest[tid * CAPACITY + i] = 0UL;
        return;
    }

    const device ulong* in_row = input + (ulong)tid * (ulong)num_cols;

    {
        uint n = (num_cols < RATE) ? num_cols : RATE;
        for (uint i = 0; i < n; i++) state[i] = in_row[i];
        for (uint i = n; i < RATE; i++) state[i] = 0UL;
        for (uint i = 0; i < CAPACITY; i++) state[RATE + i] = 0UL;
        pose2_permute_n_inplace_tg(state, C, D, W);
    }

    uint offset    = (num_cols < RATE) ? num_cols : RATE;
    uint remaining = num_cols - offset;

    while (remaining > 0u) {
        // RATE (4) == CAPACITY (4) here — state[0..CAPACITY] and
        // state[RATE..RATE+CAPACITY] are DISTINCT index ranges (the
        // upper half), so no aliasing on the rotation copy.
        for (uint i = 0; i < CAPACITY; i++) state[RATE + i] = state[i];
        uint n = (remaining < RATE) ? remaining : RATE;
        for (uint i = 0; i < n; i++)      state[i] = in_row[offset + i];
        for (uint i = n; i < RATE; i++)   state[i] = 0UL;
        pose2_permute_n_inplace_tg(state, C, D, W);
        offset    += n;
        remaining -= n;
    }

    for (uint i = 0; i < CAPACITY; i++) {
        out_digest[tid * CAPACITY + i] = gl_canonicalize(state[i]);
    }
}

// --- STEP_EVALS evmap kernel ------------------------------------------
// Ports starks.hpp::evmap's inner summation to Metal. For each of
// `nEvals` committed/custom/const polynomials, accumulates the cubic
// product LEv[opening_pos[e], k] * pol_value_at_row(k << extend_bits)
// over all k in [0, N). Reduces threadgroup-locally — one threadgroup
// per eval, EVMAP_TG_SIZE threads each sum a strided subset of rows
// and then tree-reduce in shared memory.
//
// dim==1 pols widen to [val, 0, 0]; dim==3 pols read three contiguous
// u64 at the sampled row. `buf_ids[e]` selects the backing buffer:
//   0 → aux_trace      (committed pols)
//   1 → custom_commits (per-air fixed ROM)
//   2 → const_pols_ext (const polynomials extended tree)
// Output is canonical [0, p) per component.
constant uint EVMAP_TG_SIZE = 256u;

kernel void evmap_k(
    device const ulong*  lev          [[buffer(0)]],
    device const ulong*  aux_trace    [[buffer(1)]],
    device const ulong*  custom       [[buffer(2)]],
    device const ulong*  const_pols   [[buffer(3)]],
    device const ulong*  offsets      [[buffer(4)]],
    device const ulong*  strides      [[buffer(5)]],
    device const uint*   dims         [[buffer(6)]],
    device const uint*   opening_pos  [[buffer(7)]],
    device const uint*   buf_ids      [[buffer(8)]],
    device       ulong*  evals_out    [[buffer(9)]],
    constant     uint&   N            [[buffer(10)]],
    constant     uint&   extend_bits  [[buffer(11)]],
    constant     uint&   np           [[buffer(12)]],
    threadgroup  ulong*  shmem        [[threadgroup(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    const uint eval_idx = gid;
    const uint opening  = opening_pos[eval_idx];
    const uint buf_id   = buf_ids[eval_idx];
    const ulong offset  = offsets[eval_idx];
    const ulong stride  = strides[eval_idx];
    const uint dim      = dims[eval_idx];

    device const ulong* src = (buf_id == 0u) ? aux_trace
                            : (buf_id == 1u) ? custom
                            : const_pols;

    thread ulong local_sum[3] = {0UL, 0UL, 0UL};

    for (uint k = lid; k < N; k += EVMAP_TG_SIZE) {
        const uint  row_ext = k << extend_bits;
        const ulong lev_base = (ulong)(opening + k * np) * 3UL;
        thread ulong lev_vec[3] = {
            lev[lev_base + 0UL],
            lev[lev_base + 1UL],
            lev[lev_base + 2UL]
        };

        const ulong pol_base = offset + (ulong)row_ext * stride;
        thread ulong pol_vec[3];
        if (dim == 1u) {
            pol_vec[0] = src[pol_base];
            pol_vec[1] = 0UL;
            pol_vec[2] = 0UL;
        } else {
            pol_vec[0] = src[pol_base + 0UL];
            pol_vec[1] = src[pol_base + 1UL];
            pol_vec[2] = src[pol_base + 2UL];
        }

        thread ulong prod[3];
        gl3_mul(prod, lev_vec, pol_vec);

        local_sum[0] = gl_add(local_sum[0], prod[0]);
        local_sum[1] = gl_add(local_sum[1], prod[1]);
        local_sum[2] = gl_add(local_sum[2], prod[2]);
    }

    // Stash each thread's running sum in threadgroup memory.
    shmem[lid * 3u + 0u] = local_sum[0];
    shmem[lid * 3u + 1u] = local_sum[1];
    shmem[lid * 3u + 2u] = local_sum[2];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction — works only while EVMAP_TG_SIZE is a power of two.
    for (uint s = EVMAP_TG_SIZE / 2u; s > 0u; s >>= 1) {
        if (lid < s) {
            shmem[lid*3u + 0u] = gl_add(shmem[lid*3u + 0u], shmem[(lid + s)*3u + 0u]);
            shmem[lid*3u + 1u] = gl_add(shmem[lid*3u + 1u], shmem[(lid + s)*3u + 1u]);
            shmem[lid*3u + 2u] = gl_add(shmem[lid*3u + 2u], shmem[(lid + s)*3u + 2u]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0u) {
        evals_out[eval_idx * 3u + 0u] = gl_canonicalize(shmem[0]);
        evals_out[eval_idx * 3u + 1u] = gl_canonicalize(shmem[1]);
        evals_out[eval_idx * 3u + 2u] = gl_canonicalize(shmem[2]);
    }
}
)MSL";

// Compile kMetalSrc once per context, cache the resulting MTLLibrary on
// the context. Subsequent calls are lock-free on the cached pointer.
id<MTLLibrary> get_or_compile_library(Context* ctx) {
    if (ctx->cached_library != nil) return ctx->cached_library;
    std::lock_guard<std::mutex> lock(ctx->pso_mutex);
    if (ctx->cached_library != nil) return ctx->cached_library;
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kMetalSrc];
    id<MTLLibrary> lib = [ctx->device newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        NSString* msg = err ? [err localizedDescription] : @"(no error info)";
        std::fprintf(stderr, "pil2::metal: MSL compile failed: %s\n", [msg UTF8String]);
        std::abort();
    }
    ctx->cached_library = lib;
    return lib;
}

// Returns a cached MTLComputePipelineState for the named kernel, creating
// (and caching) one on cache-miss. PSO construction does real compilation
// (~few ms); caching saves that on every subsequent dispatch of the same
// kernel. Mutex-protected — dispatches are serial today but thread-safe
// future-proofing is cheap.
id<MTLComputePipelineState> get_or_make_pso(Context* ctx, NSString* kernel_name) {
    const char* c_name = [kernel_name UTF8String];
    std::string key(c_name);
    {
        std::lock_guard<std::mutex> lock(ctx->pso_mutex);
        auto it = ctx->pso_cache.find(key);
        if (it != ctx->pso_cache.end()) return it->second;
    }
    id<MTLLibrary> lib = get_or_compile_library(ctx);
    id<MTLFunction> fn = [lib newFunctionWithName:kernel_name];
    if (!fn) {
        std::fprintf(stderr, "pil2::metal: kernel not found: %s\n", c_name);
        std::abort();
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [ctx->device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        NSString* msg = err ? [err localizedDescription] : @"(no error info)";
        std::fprintf(stderr, "pil2::metal: PSO create failed for %s: %s\n",
                     c_name, [msg UTF8String]);
        std::abort();
    }
    std::lock_guard<std::mutex> lock(ctx->pso_mutex);
    ctx->pso_cache[key] = pso;
    return pso;
}

} // namespace

void run_field_op(ContextHandle   ctx,
                  const char*     op,
                  const uint64_t* a,
                  const uint64_t* b,
                  uint64_t*       out,
                  uint32_t        n) {
    if (!ctx) { fatal("run_field_op: null context"); }
    if (n == 0) { return; }

    const char* kernel_name = nullptr;
    if      (std::strcmp(op, "add") == 0) kernel_name = "field_add_k";
    else if (std::strcmp(op, "sub") == 0) kernel_name = "field_sub_k";
    else if (std::strcmp(op, "mul") == 0) kernel_name = "field_mul_k";
    else { fatal("run_field_op: unknown op (want add|sub|mul)"); }

    @autoreleasepool {
        id<MTLDevice>       dev = ctx->device;
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];

        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:kMetalSrc];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            NSString* msg = err ? [err localizedDescription] : @"(no error info)";
            std::fprintf(stderr, "pil2::metal: field MSL compile failed: %s\n", [msg UTF8String]);
            std::abort();
        }

        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kernel_name]];
        if (!fn) { fatal("run_field_op: kernel not found in library"); }

        err = nil;
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            NSString* msg = err ? [err localizedDescription] : @"(no error info)";
            std::fprintf(stderr, "pil2::metal: field PSO create failed: %s\n", [msg UTF8String]);
            std::abort();
        }

        const size_t bytes = static_cast<size_t>(n) * sizeof(uint64_t);
        id<MTLBuffer> bufA = [dev newBufferWithBytes:a  length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [dev newBufferWithBytes:b  length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufO = [dev newBufferWithLength:bytes          options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufO) { fatal("run_field_op: buffer alloc failed"); }

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufO offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(n) atIndex:3];

        const NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: field cmd buffer error: %s\n", [msg UTF8String]);
            std::abort();
        }

        std::memcpy(out, [bufO contents], bytes);
    }
}

// Test helper that dispatches one of the gl3_*_k kernels over `n` cubic-ext
// elements. Mirrors run_field_op — used only by the gl3 unit tests.
// Layout: a and out are n*3 u64; b is either n*3 u64 (for add/sub/mul) or
// n u64 (for mul_scalar).
void run_gl3_op(ContextHandle   ctx,
                const char*     op,
                const uint64_t* a,
                const uint64_t* b,
                uint64_t*       out,
                uint32_t        n) {
    if (!ctx) { fatal("run_gl3_op: null context"); }
    if (n == 0) { return; }

    const char* kernel_name = nullptr;
    bool b_is_scalar = false;
    if      (std::strcmp(op, "add")        == 0) kernel_name = "gl3_add_k";
    else if (std::strcmp(op, "sub")        == 0) kernel_name = "gl3_sub_k";
    else if (std::strcmp(op, "mul")        == 0) kernel_name = "gl3_mul_k";
    else if (std::strcmp(op, "mul_scalar") == 0) { kernel_name = "gl3_mul_scalar_k"; b_is_scalar = true; }
    else { fatal("run_gl3_op: unknown op (want add|sub|mul|mul_scalar)"); }

    @autoreleasepool {
        id<MTLDevice>       dev = ctx->device;
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];

        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:kMetalSrc];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            NSString* msg = err ? [err localizedDescription] : @"(no error info)";
            std::fprintf(stderr, "pil2::metal: gl3 MSL compile failed: %s\n", [msg UTF8String]);
            std::abort();
        }
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kernel_name]];
        if (!fn) { fatal("run_gl3_op: kernel not found"); }
        err = nil;
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { fatal("run_gl3_op: PSO create failed"); }

        const size_t out_bytes = static_cast<size_t>(n) * 3 * sizeof(uint64_t);
        const size_t a_bytes   = out_bytes;
        const size_t b_bytes   = b_is_scalar
            ? static_cast<size_t>(n) * sizeof(uint64_t)
            : out_bytes;

        id<MTLBuffer> bufA = [dev newBufferWithBytes:a length:a_bytes   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [dev newBufferWithBytes:b length:b_bytes   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufO = [dev newBufferWithLength:out_bytes         options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufO) { fatal("run_gl3_op: buffer alloc failed"); }

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufO offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        const NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: gl3 cmd error: %s\n", [msg UTF8String]);
            std::abort();
        }
        std::memcpy(out, [bufO contents], out_bytes);
    }
}

// Production gl3 element-wise multiply. Unlike run_gl3_op above:
//   - PSO is fetched from the process cache (get_or_make_pso), so there's
//     no per-call source compile or library build.
//   - a / b / out are looked up via metal_resolve_shared and bound at the
//     resolved offset when registered → zero copy. Non-registered pointers
//     fall back to the persistent scratch pool + memcpy; that fallback is
//     cheap the first time (slot-sized alloc amortised) and free on reuse.
//   - Dispatched on the per-thread stream queue, so concurrent proof
//     workers don't serialise through the shared queue.
//
// Layout: a, b are each n * 3 u64 (dense, stride = 3). When dst_stride == 3,
// the dense gl3_mul_k kernel runs and out is a contiguous n * 3 u64 span.
// When dst_stride > 3, gl3_mul_strided_k runs, writing
// out[tid * dst_stride + c] for c in [0, 3); the strict-inequality cells
// between c == 3 and c == dst_stride are preserved (imPol writing into a
// wider cm-section row). The resolver-miss fallback below keeps that
// invariant by copying only the kernel-written cells back to `out`.
void run_gl3_mul(ContextHandle   ctx,
                 const uint64_t* a,
                 const uint64_t* b,
                 uint64_t*       out,
                 uint32_t        n,
                 uint32_t        dst_stride) {
    if (!ctx) { fatal("run_gl3_mul: null context"); }
    if (n == 0) { return; }
    if (!a || !b || !out) { fatal("run_gl3_mul: null pointer"); }
    if (dst_stride < 3u) { fatal("run_gl3_mul: dst_stride must be >= 3"); }

    @autoreleasepool {
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];
        const bool strided = (dst_stride > 3u);
        id<MTLComputePipelineState> pso = get_or_make_pso(
            ctx, strided ? @"gl3_mul_strided_k" : @"gl3_mul_k");

        // Src is always dense. Dst span covers the last-touched cell:
        // tid*(n-1) * dst_stride + 3 u64s.
        const size_t src_bytes = static_cast<size_t>(n) * 3u * sizeof(uint64_t);
        const size_t dst_bytes = (static_cast<size_t>(n - 1) * dst_stride + 3u)
                                   * sizeof(uint64_t);

        id<MTLBuffer> bufA = nil; size_t offA = 0;
        if (!metal_resolve_shared(ctx, a, src_bytes, &bufA, &offA)) {
            bufA = scratch_borrow(ctx, kGl3MulSrcA, src_bytes);
            std::memcpy([bufA contents], a, src_bytes);
            offA = 0;
        }

        id<MTLBuffer> bufB = nil; size_t offB = 0;
        if (!metal_resolve_shared(ctx, b, src_bytes, &bufB, &offB)) {
            bufB = scratch_borrow(ctx, kGl3MulSrcB, src_bytes);
            std::memcpy([bufB contents], b, src_bytes);
            offB = 0;
        }

        id<MTLBuffer> bufO = nil; size_t offO = 0;
        const bool outResolved = metal_resolve_shared(ctx, out, dst_bytes, &bufO, &offO);
        if (!outResolved) {
            bufO = scratch_borrow(ctx, kGl3MulDst, dst_bytes);
            offO = 0;
        }

        if (!bufA || !bufB || !bufO) { fatal("run_gl3_mul: buffer alloc failed"); }

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:offA atIndex:0];
        [enc setBuffer:bufB offset:offB atIndex:1];
        [enc setBuffer:bufO offset:offO atIndex:2];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        if (strided) {
            [enc setBytes:&dst_stride length:sizeof(dst_stride) atIndex:4];
        }
        const NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: gl3_mul cmd error: %s\n", [msg UTF8String]);
            std::abort();
        }
        if (!outResolved) {
            if (strided) {
                // Kernel wrote only the 3 cells at tid*stride. A dense
                // memcpy of dst_bytes would stomp the between-row gap
                // cells (other imPols sharing the cm-section row). Copy
                // only the cells the kernel actually wrote.
                const uint64_t* src = static_cast<const uint64_t*>([bufO contents]);
                for (uint32_t t = 0; t < n; ++t) {
                    const size_t base = static_cast<size_t>(t) * dst_stride;
                    out[base + 0] = src[base + 0];
                    out[base + 1] = src[base + 1];
                    out[base + 2] = src[base + 2];
                }
            } else {
                std::memcpy(out, [bufO contents], src_bytes);
            }
        }
    }
}

// Unit-test bridge: applies gl_inv element-wise over `a` (n u64s) and
// writes canonical result into `out`. Caller must ensure a[i] != 0 for
// every element (Fermat's a^(p-2) for a==0 returns 0, which mirrors the
// only way to observe incorrect inverse math in downstream use).
void run_gl_inv_test(ContextHandle   ctx,
                     const uint64_t* a,
                     uint64_t*       out,
                     uint32_t        n) {
    if (!ctx) { fatal("run_gl_inv_test: null context"); }
    if (n == 0) { return; }
    @autoreleasepool {
        id<MTLDevice>       dev = ctx->device;
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];
        id<MTLComputePipelineState> pso = get_or_make_pso(ctx, @"gl_inv_k");

        const size_t bytes = static_cast<size_t>(n) * sizeof(uint64_t);
        id<MTLBuffer> bufA = [dev newBufferWithBytes:a  length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufO = [dev newBufferWithLength:bytes          options:MTLResourceStorageModeShared];
        if (!bufA || !bufO) { fatal("run_gl_inv_test: buffer alloc failed"); }

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufO offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        const NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: gl_inv_test cmd error: %s\n", [msg UTF8String]);
            std::abort();
        }
        std::memcpy(out, [bufO contents], bytes);
    }
}

// Unit-test bridge: cubic inverse element-wise. `a` and `out` are each
// n*3 u64s. Caller contract: a[i] is not the zero cubic element.
void run_gl3_inv_test(ContextHandle   ctx,
                      const uint64_t* a,
                      uint64_t*       out,
                      uint32_t        n) {
    if (!ctx) { fatal("run_gl3_inv_test: null context"); }
    if (n == 0) { return; }
    @autoreleasepool {
        id<MTLDevice>       dev = ctx->device;
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];
        id<MTLComputePipelineState> pso = get_or_make_pso(ctx, @"gl3_inv_k");

        const size_t bytes = static_cast<size_t>(n) * 3 * sizeof(uint64_t);
        id<MTLBuffer> bufA = [dev newBufferWithBytes:a  length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufO = [dev newBufferWithLength:bytes          options:MTLResourceStorageModeShared];
        if (!bufA || !bufO) { fatal("run_gl3_inv_test: buffer alloc failed"); }

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufO offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        const NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: gl3_inv_test cmd error: %s\n", [msg UTF8String]);
            std::abort();
        }
        std::memcpy(out, [bufO contents], bytes);
    }
}

// STEP_EVALS evmap — computes `evals_out[e] = sum_{k in [0,N)} LEv[opening_pos[e], k]
// · pol_at_row(k << extend_bits)` per eval. Mirrors starks.hpp::evmap.
// Backing buffers (aux_trace, custom, const_pols) may be null when no
// eval in the set references that type; a dummy 1-u64 Metal buffer keeps
// the binding valid for the kernel.
void run_evmap_metal(ContextHandle    ctx,
                     const uint64_t*  lev,
                     const uint64_t*  aux_trace,
                     const uint64_t*  custom,
                     const uint64_t*  const_pols,
                     const uint64_t*  offsets,
                     const uint64_t*  strides,
                     const uint32_t*  dims,
                     const uint32_t*  opening_pos,
                     const uint32_t*  buf_ids,
                     uint64_t*        evals_out,
                     uint32_t         N,
                     uint32_t         extend_bits,
                     uint32_t         np,
                     uint32_t         n_evals,
                     uint32_t         lev_len_u64,
                     uint32_t         aux_trace_len_u64,
                     uint32_t         custom_len_u64,
                     uint32_t         const_pols_len_u64) {
    if (!ctx) { fatal("run_evmap_metal: null context"); }
    if (n_evals == 0) { return; }
    if (N == 0)       { fatal("run_evmap_metal: N must be > 0"); }
    if (np == 0)      { fatal("run_evmap_metal: np must be > 0"); }

    @autoreleasepool {
        id<MTLDevice>       dev = ctx->device;
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];
        id<MTLComputePipelineState> pso = get_or_make_pso(ctx, @"evmap_k");

        const size_t lev_bytes       = static_cast<size_t>(lev_len_u64)        * sizeof(uint64_t);
        const size_t aux_bytes       = static_cast<size_t>(aux_trace_len_u64)  * sizeof(uint64_t);
        const size_t custom_bytes    = static_cast<size_t>(custom_len_u64)     * sizeof(uint64_t);
        const size_t const_pol_bytes = static_cast<size_t>(const_pols_len_u64) * sizeof(uint64_t);
        const size_t tbl_u64_bytes   = static_cast<size_t>(n_evals) * sizeof(uint64_t);
        const size_t tbl_u32_bytes   = static_cast<size_t>(n_evals) * sizeof(uint32_t);
        const size_t out_bytes       = static_cast<size_t>(n_evals) * 3 * sizeof(uint64_t);

        // B.18 unified-memory fast path: if the caller's buffers live in
        // a registered Metal allocation, bind that buffer zero-copy with
        // the resolved offset. Falls back to the scratch pool + memcpy
        // when the caller's buffer isn't registered (test harness,
        // transient LEv, etc).
        auto resolve_or_borrow = [&](const uint64_t* p, size_t bytes, int slot,
                                     bool cached,
                                     size_t* out_offset) -> id<MTLBuffer> {
            *out_offset = 0;
            if (p == nullptr || bytes == 0) {
                return [dev newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];
            }
            id<MTLBuffer> resolved = nil;
            size_t resolved_off = 0;
            if (metal_resolve_shared(ctx, p, bytes, &resolved, &resolved_off)) {
                *out_offset = resolved_off;
                return resolved;
            }
            if (cached) {
                return scratch_upload_cached(ctx, slot, p, bytes);
            }
            id<MTLBuffer> b = scratch_borrow(ctx, slot, bytes);
            std::memcpy([b contents], p, bytes);
            return b;
        };
        size_t offLev = 0, offAux = 0, offCust = 0, offConst = 0;
        id<MTLBuffer> bufLev   = resolve_or_borrow(lev,        lev_bytes,       kEvmapSlotLev,    /*cached=*/false, &offLev);
        id<MTLBuffer> bufAux   = resolve_or_borrow(aux_trace,  aux_bytes,       kEvmapSlotAux,    /*cached=*/false, &offAux);
        id<MTLBuffer> bufCust  = resolve_or_borrow(custom,     custom_bytes,    kEvmapSlotCustom, /*cached=*/true,  &offCust);
        id<MTLBuffer> bufConst = resolve_or_borrow(const_pols, const_pol_bytes, kEvmapSlotConst,  /*cached=*/true,  &offConst);
        id<MTLBuffer> bufOff   = [dev newBufferWithBytes:offsets     length:tbl_u64_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufStr   = [dev newBufferWithBytes:strides     length:tbl_u64_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDim   = [dev newBufferWithBytes:dims        length:tbl_u32_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOp    = [dev newBufferWithBytes:opening_pos length:tbl_u32_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufBuf   = [dev newBufferWithBytes:buf_ids     length:tbl_u32_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut   = [dev newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufLev   offset:offLev   atIndex:0];
        [enc setBuffer:bufAux   offset:offAux   atIndex:1];
        [enc setBuffer:bufCust  offset:offCust  atIndex:2];
        [enc setBuffer:bufConst offset:offConst atIndex:3];
        [enc setBuffer:bufOff   offset:0 atIndex:4];
        [enc setBuffer:bufStr   offset:0 atIndex:5];
        [enc setBuffer:bufDim   offset:0 atIndex:6];
        [enc setBuffer:bufOp    offset:0 atIndex:7];
        [enc setBuffer:bufBuf   offset:0 atIndex:8];
        [enc setBuffer:bufOut   offset:0 atIndex:9];
        [enc setBytes:&N           length:sizeof(N)           atIndex:10];
        [enc setBytes:&extend_bits length:sizeof(extend_bits) atIndex:11];
        [enc setBytes:&np          length:sizeof(np)          atIndex:12];
        // Threadgroup memory: 256 threads × 3 u64 = 6KB.
        [enc setThreadgroupMemoryLength:256 * 3 * sizeof(uint64_t) atIndex:0];

        const NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
        // One threadgroup per eval.
        [enc dispatchThreadgroups:MTLSizeMake(n_evals, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: evmap cmd error: %s\n", [msg UTF8String]);
            std::abort();
        }
        std::memcpy(evals_out, [bufOut contents], out_bytes);
    }
}

void fri_fold_w8_metal(ContextHandle   ctx,
                       uint64_t*       pol,
                       const uint64_t* challenge,
                       uint64_t        pol2N,
                       uint64_t        polShiftInv,
                       uint64_t        wi,
                       uint64_t        inv8,
                       const uint64_t* roots8) {
    if (!ctx) { fatal("fri_fold_w8_metal: null context"); }
    if (pol2N == 0) { return; }
    constexpr uint64_t nX = 8;

    @autoreleasepool {
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];
        id<MTLComputePipelineState> pso = get_or_make_pso(ctx, @"fri_fold_w8_k");

        const size_t pol_bytes = static_cast<size_t>(nX * pol2N) * 3 * sizeof(uint64_t);
        const size_t chal_bytes  = 3 * sizeof(uint64_t);
        const size_t roots_bytes = nX * sizeof(uint64_t);

        // B.19 zero-copy: if pol lives in a Metal-registered buffer
        // (typical case: aux_trace + mapOffsets["f", true] offset_f),
        // bind that buffer directly. Otherwise fall back to scratch +
        // memcpy. pol is by far the largest input/output here — up to
        // 192MB on step 1 of the main fibonacci-square AIR — so the
        // zero-copy path is the big win on this function.
        id<MTLBuffer> bufPol = nil;
        size_t polOffset = 0;
        bool polResolved = metal_resolve_shared(ctx, pol, pol_bytes, &bufPol, &polOffset);
        if (!polResolved) {
            bufPol = scratch_borrow(ctx, 0, pol_bytes);
            std::memcpy([bufPol contents], pol, pol_bytes);
            polOffset = 0;
        }
        id<MTLBuffer> bufChal = scratch_borrow(ctx, 1, chal_bytes);
        id<MTLBuffer> bufRoot = scratch_borrow(ctx, 2, roots_bytes);
        std::memcpy([bufChal contents], challenge, chal_bytes);
        std::memcpy([bufRoot contents], roots8,    roots_bytes);

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufPol  offset:polOffset atIndex:0];
        [enc setBuffer:bufChal offset:0 atIndex:1];
        [enc setBuffer:bufRoot offset:0 atIndex:2];
        uint32_t n = static_cast<uint32_t>(pol2N);
        [enc setBytes:&n           length:sizeof(n)           atIndex:3];
        [enc setBytes:&polShiftInv length:sizeof(polShiftInv) atIndex:4];
        [enc setBytes:&wi          length:sizeof(wi)          atIndex:5];
        [enc setBytes:&inv8        length:sizeof(inv8)        atIndex:6];
        const NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 64);
        [enc dispatchThreads:MTLSizeMake(pol2N, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: fri_fold cmd error: %s\n",
                         [msg UTF8String]);
            std::abort();
        }
        // Result lives in the first pol2N * 3 u64s; leave the rest alone.
        if (!polResolved) {
            std::memcpy(pol, [bufPol contents], static_cast<size_t>(pol2N) * 3 * sizeof(uint64_t));
        }
        // When resolved, the kernel wrote directly into the caller's
        // shared-memory buffer (aux_trace region) — nothing to copy.
    }
}

void run_op_dispatch(ContextHandle   ctx,
                     const char*     flavor,
                     uint32_t        op_code,
                     const uint64_t* a,
                     const uint64_t* b,
                     uint64_t*       out,
                     uint32_t        n) {
    if (!ctx) { fatal("run_op_dispatch: null context"); }
    if (n == 0) { return; }

    NSString* kernel_name;
    size_t a_bytes, b_bytes, out_bytes;
    if      (std::strcmp(flavor, "gl_op") == 0) {
        kernel_name = @"gl_op_k";
        a_bytes = out_bytes = (size_t)n * sizeof(uint64_t);
        b_bytes = a_bytes;
    } else if (std::strcmp(flavor, "gl3_op") == 0) {
        kernel_name = @"gl3_op_k";
        a_bytes = out_bytes = (size_t)n * 3 * sizeof(uint64_t);
        b_bytes = a_bytes;
    } else if (std::strcmp(flavor, "gl3_op_31") == 0) {
        kernel_name = @"gl3_op_31_k";
        a_bytes = out_bytes = (size_t)n * 3 * sizeof(uint64_t);
        b_bytes = (size_t)n * sizeof(uint64_t);
    } else {
        fatal("run_op_dispatch: unknown flavor (want gl_op|gl3_op|gl3_op_31)");
        return;
    }

    @autoreleasepool {
        id<MTLDevice>       dev = ctx->device;
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];
        id<MTLComputePipelineState> pso = get_or_make_pso(ctx, kernel_name);

        id<MTLBuffer> bufA = [dev newBufferWithBytes:a length:a_bytes   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [dev newBufferWithBytes:b length:b_bytes   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufO = [dev newBufferWithLength:out_bytes         options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufO) { fatal("run_op_dispatch: buffer alloc failed"); }

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufO offset:0 atIndex:2];
        [enc setBytes:&n       length:sizeof(n)       atIndex:3];
        [enc setBytes:&op_code length:sizeof(op_code) atIndex:4];
        const NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: op_dispatch cmd error: %s\n", [msg UTF8String]);
            std::abort();
        }
        std::memcpy(out, [bufO contents], out_bytes);
    }
}

void run_expr_vm_min(ContextHandle   ctx,
                     const uint8_t*  ops,
                     const uint16_t* args,
                     const uint64_t* numbers,
                     const uint64_t* trace,
                     const uint64_t* aux_trace,
                     const uint64_t* const_pols,
                     const uint32_t* stage_offsets,
                     const uint32_t* stage_ncols,
                     const int64_t*  next_strides,
                     uint64_t*       dst,
                     uint32_t        n_ops,
                     uint32_t        n_args,
                     uint32_t        n_numbers,
                     uint32_t        trace_len_u64,
                     uint32_t        aux_trace_len_u64,
                     uint32_t        const_pols_len_u64,
                     uint32_t        n_stages_plus_2,
                     uint32_t        next_strides_len,
                     uint32_t        n_threads,
                     uint32_t        domain_size,
                     uint32_t        buffer_commits_size,
                     bool            domain_extended,
                     const ExprVmFlatTables& flat,
                     uint32_t        dest_dim,
                     const ExprVmCustomCommits& custom,
                     const ExprVmProverHelpers& prover_helpers,
                     bool            dest_inverse,
                     uint32_t        dst_stride) {
    if (!ctx)           { fatal("run_expr_vm_min: null context"); }
    if (n_threads == 0) { return; }
    if (n_ops == 0)     { fatal("run_expr_vm_min: n_ops must be > 0"); }
    if (n_stages_plus_2 == 0) { fatal("run_expr_vm_min: n_stages_plus_2 must be > 0"); }
    if (next_strides_len == 0) { fatal("run_expr_vm_min: next_strides_len must be > 0"); }
    if (domain_size == 0 || (domain_size & (domain_size - 1u)) != 0) {
        fatal("run_expr_vm_min: domain_size must be a power of two");
    }
    if (dest_dim != 1u && dest_dim != 3u) {
        fatal("run_expr_vm_min: dest_dim must be 1 (base field) or 3 (cubic)");
    }
    if (dst_stride == 0u) {
        // Sentinel: caller didn't set a stride, use dense layout.
        dst_stride = dest_dim;
    }
    if (dst_stride < dest_dim) {
        fatal("run_expr_vm_min: dst_stride must be >= dest_dim");
    }
    if (dest_inverse && dest_dim != 3u) {
        fatal("run_expr_vm_min: dest_inverse currently only supported for dest_dim=3");
    }
    if (custom.count > buffer_commits_size) {
        fatal("run_expr_vm_min: custom.count must be <= buffer_commits_size");
    }
    const uint32_t custom_lo_bound = buffer_commits_size - custom.count;

    @autoreleasepool {
        id<MTLDevice>       dev = ctx->device;
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];
        id<MTLComputePipelineState> pso = get_or_make_pso(ctx, @"expr_vm_min_k");

        const size_t ops_bytes       = static_cast<size_t>(n_ops)           * sizeof(uint8_t);
        const size_t args_bytes      = static_cast<size_t>(n_args)          * sizeof(uint16_t);
        const size_t numbers_bytes   = static_cast<size_t>(n_numbers)       * sizeof(uint64_t);
        const size_t trace_bytes     = static_cast<size_t>(trace_len_u64)     * sizeof(uint64_t);
        const size_t aux_bytes       = static_cast<size_t>(aux_trace_len_u64) * sizeof(uint64_t);
        const size_t const_pol_bytes = static_cast<size_t>(const_pols_len_u64)* sizeof(uint64_t);
        const size_t stage_tab_bytes = static_cast<size_t>(n_stages_plus_2)   * sizeof(uint32_t);
        // When the kernel writes strided (dst_stride > dest_dim for
        // imPol-style writes into a wider cm-section row), the kernel
        // writes at `tid * dst_stride + c` for c < dest_dim. The
        // farthest write is at `(n_threads - 1) * dst_stride + (dest_dim - 1)`,
        // so the minimum range covering all writes is
        // `(n_threads - 1) * dst_stride + dest_dim` u64s. Using this tight
        // size (vs the loose `n_threads * dst_stride`) lets the resolver
        // succeed when the caller's `dst + loose` would overshoot the
        // registered allocation by at most `dst_stride - dest_dim` u64s
        // — which is exactly the fibonacci-square imPol pattern.
        // For the dense case (stride == dim) the formula reduces to
        // n_threads * dest_dim, matching the old value.
        const size_t dst_bytes       = (static_cast<size_t>(n_threads - 1) * dst_stride + dest_dim) * sizeof(uint64_t);

        // Metal refuses a nil / zero-length buffer binding. Programs that
        // don't reference a given source pass nullptr + length 0; we back
        // the binding with a 1-unit dummy, and the kernel simply never
        // reads it because its type-branch goes elsewhere.
        id<MTLBuffer> bufOps     = [dev newBufferWithBytes:ops  length:ops_bytes  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufArgs    = [dev newBufferWithBytes:args length:args_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufNumbers = (n_numbers > 0)
            ? [dev newBufferWithBytes:numbers length:numbers_bytes options:MTLResourceStorageModeShared]
            : [dev newBufferWithLength:sizeof(uint64_t)            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufTrace   = (trace != nullptr && trace_bytes > 0)
            ? [dev newBufferWithBytes:trace length:trace_bytes options:MTLResourceStorageModeShared]
            : [dev newBufferWithLength:sizeof(uint64_t)        options:MTLResourceStorageModeShared];

        // B.18 unified-memory fast paths: if the caller's buffer sits
        // inside a buffer allocated through metal_alloc_shared, bind
        // that MTLBuffer directly with the resolved offset — no memcpy.
        // Falls through to the scratch pool + memcpy otherwise.
        id<MTLBuffer> bufAux = nil;
        size_t bufAuxOffset = 0;
        if (aux_trace != nullptr && aux_bytes > 0) {
            if (!metal_resolve_shared(ctx, aux_trace, aux_bytes, &bufAux, &bufAuxOffset)) {
                bufAux = scratch_borrow(ctx, kVmSlotAuxTrace, aux_bytes);
                std::memcpy([bufAux contents], aux_trace, aux_bytes);
                bufAuxOffset = 0;
            }
        } else {
            bufAux = [dev newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> bufConst = nil;
        size_t bufConstOffset = 0;
        if (const_pols != nullptr && const_pol_bytes > 0) {
            if (!metal_resolve_shared(ctx, const_pols, const_pol_bytes, &bufConst, &bufConstOffset)) {
                // const_pols is write-once per AIR setup — same content
                // across every dispatch. The cache skips the memcpy when
                // the same pointer is passed again.
                bufConst = scratch_upload_cached(ctx, kVmSlotConstPols, const_pols, const_pol_bytes);
                bufConstOffset = 0;
            }
        } else {
            bufConst = [dev newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];
        }

        // B.5 flat-constant tables. Each is nullable; a 1-u64 dummy keeps
        // the binding valid when the bytecode doesn't reference it.
        auto make_flat_buf = [&](const uint64_t* p, uint32_t len_u64) -> id<MTLBuffer> {
            if (p != nullptr && len_u64 > 0) {
                return [dev newBufferWithBytes:p
                                        length:static_cast<size_t>(len_u64) * sizeof(uint64_t)
                                       options:MTLResourceStorageModeShared];
            }
            return [dev newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> bufPublic     = make_flat_buf(flat.public_inputs,   flat.public_inputs_len_u64);
        id<MTLBuffer> bufAirVals    = make_flat_buf(flat.air_values,      flat.air_values_len_u64);
        id<MTLBuffer> bufProofVals  = make_flat_buf(flat.proof_values,    flat.proof_values_len_u64);
        id<MTLBuffer> bufGroupVals  = make_flat_buf(flat.airgroup_values, flat.airgroup_values_len_u64);
        id<MTLBuffer> bufChallenges = make_flat_buf(flat.challenges,      flat.challenges_len_u64);
        id<MTLBuffer> bufEvals      = make_flat_buf(flat.evals,           flat.evals_len_u64);

        id<MTLBuffer> bufStageOff = [dev newBufferWithBytes:stage_offsets length:stage_tab_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufStageNc  = [dev newBufferWithBytes:stage_ncols   length:stage_tab_bytes options:MTLResourceStorageModeShared];
        const size_t strides_bytes = static_cast<size_t>(next_strides_len) * sizeof(int64_t);
        id<MTLBuffer> bufStrides  = [dev newBufferWithBytes:next_strides length:strides_bytes options:MTLResourceStorageModeShared];

        // Custom commits: dummy 1-unit buffer when count == 0 (bytecode
        // never references the range, but Metal still needs valid bindings).
        id<MTLBuffer> bufCustomData, bufCustomOff, bufCustomNc;
        if (custom.count > 0 && custom.data != nullptr && custom.data_len_u64 > 0) {
            bufCustomData = [dev newBufferWithBytes:custom.data
                                             length:static_cast<size_t>(custom.data_len_u64) * sizeof(uint64_t)
                                            options:MTLResourceStorageModeShared];
            bufCustomOff  = [dev newBufferWithBytes:custom.offsets
                                             length:static_cast<size_t>(custom.count) * sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
            bufCustomNc   = [dev newBufferWithBytes:custom.ncols
                                             length:static_cast<size_t>(custom.count) * sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
        } else {
            bufCustomData = [dev newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];
            bufCustomOff  = [dev newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
            bufCustomNc   = [dev newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        }

        // ProverHelpers: two optional dim-1 buffers (x / x_n chosen by caller;
        // zi concatenated per boundary). Dummy 1-unit fallback when null.
        id<MTLBuffer> bufX  = (prover_helpers.x_current != nullptr && prover_helpers.x_current_len_u64 > 0)
            ? [dev newBufferWithBytes:prover_helpers.x_current
                               length:static_cast<size_t>(prover_helpers.x_current_len_u64) * sizeof(uint64_t)
                              options:MTLResourceStorageModeShared]
            : [dev newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufZi = (prover_helpers.zi != nullptr && prover_helpers.zi_len_u64 > 0)
            ? [dev newBufferWithBytes:prover_helpers.zi
                               length:static_cast<size_t>(prover_helpers.zi_len_u64) * sizeof(uint64_t)
                              options:MTLResourceStorageModeShared]
            : [dev newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufXis = (prover_helpers.xis != nullptr && prover_helpers.xis_len_u64 > 0)
            ? [dev newBufferWithBytes:prover_helpers.xis
                               length:static_cast<size_t>(prover_helpers.xis_len_u64) * sizeof(uint64_t)
                              options:MTLResourceStorageModeShared]
            : [dev newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];

        // dst — resolve to a registered MTLBuffer if the caller's dst
        // pointer lives in one (the normal case for unified memory
        // aux_trace: dst is aux_trace + offset_f). If not resolved,
        // fall back to the scratch pool and memcpy back after dispatch.
        // Note: when the helper-level strided-writeback path is active
        // it passes a heap-allocated scratch as dst, which is never
        // in the registry — so the fallback naturally triggers.
        id<MTLBuffer> bufDst = nil;
        size_t bufDstOffset = 0;
        bool dstResolvedShared = false;
        if (metal_resolve_shared(ctx, dst, dst_bytes, &bufDst, &bufDstOffset)) {
            dstResolvedShared = true;
        } else {
            bufDst = scratch_borrow(ctx, kVmSlotDst, dst_bytes);
            bufDstOffset = 0;
            // Diagnostic: resolver-miss + strided writes together are
            // dangerous — the kernel writes sparsely (only tid*stride+c
            // for c<dim) into scratch, but the post-kernel memcpy below
            // copies the DENSE dst_bytes range back onto `dst`,
            // clobbering gap cells with scratch garbage. Log it so we
            // can confirm whether production ever hits this path.
            if (dst_stride > dest_dim) {
                static const bool kDiag = []{
                    const char* e = std::getenv("PIL2_METAL_VM_STRIDED_MISS_DIAG");
                    return e && e[0] == '1';
                }();
                if (kDiag) {
                    const void* base = nullptr;
                    size_t sz = 0, off = 0;
                    if (metal_shared_containing(ctx, dst, &base, &sz, &off)) {
                        const size_t overshoot = (off + dst_bytes > sz) ? (off + dst_bytes - sz) : 0;
                        std::fprintf(stderr,
                            "[vm] STRIDED MISS dst=%p n=%u stride=%u dim=%u dst_bytes=%zu | alloc_base=%p alloc_size=%zu offset_in_alloc=%zu overshoot=%zu\n",
                            dst, n_threads, dst_stride, dest_dim, dst_bytes,
                            base, sz, off, overshoot);
                    } else {
                        std::fprintf(stderr,
                            "[vm] STRIDED MISS dst=%p n=%u stride=%u dim=%u dst_bytes=%zu | NOT in any shared allocation\n",
                            dst, n_threads, dst_stride, dest_dim, dst_bytes);
                    }
                }
            }
        }
        if (!bufOps || !bufArgs || !bufNumbers || !bufTrace || !bufAux || !bufConst
            || !bufPublic || !bufAirVals || !bufProofVals || !bufGroupVals
            || !bufChallenges || !bufEvals
            || !bufStageOff || !bufStageNc || !bufStrides
            || !bufCustomData || !bufCustomOff || !bufCustomNc
            || !bufX || !bufZi || !bufXis
            || !bufDst) {
            fatal("run_expr_vm_min: buffer alloc failed");
        }

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufOps      offset:0 atIndex:0];
        [enc setBuffer:bufArgs     offset:0 atIndex:1];
        [enc setBuffer:bufNumbers  offset:0 atIndex:2];
        [enc setBuffer:bufDst      offset:bufDstOffset atIndex:3];
        [enc setBytes:&n_ops               length:sizeof(n_ops)               atIndex:4];
        [enc setBytes:&n_threads           length:sizeof(n_threads)           atIndex:5];
        [enc setBytes:&buffer_commits_size length:sizeof(buffer_commits_size) atIndex:6];
        [enc setBuffer:bufTrace    offset:0 atIndex:7];
        [enc setBuffer:bufAux      offset:bufAuxOffset atIndex:8];
        [enc setBuffer:bufStageOff offset:0 atIndex:9];
        [enc setBuffer:bufStageNc  offset:0 atIndex:10];
        uint8_t de = domain_extended ? 1u : 0u;
        [enc setBytes:&de length:sizeof(de) atIndex:11];
        [enc setBuffer:bufConst       offset:bufConstOffset atIndex:12];
        [enc setBuffer:bufPublic      offset:0 atIndex:13];
        [enc setBuffer:bufAirVals     offset:0 atIndex:14];
        [enc setBuffer:bufProofVals   offset:0 atIndex:15];
        [enc setBuffer:bufGroupVals   offset:0 atIndex:16];
        [enc setBuffer:bufChallenges  offset:0 atIndex:17];
        [enc setBuffer:bufEvals       offset:0 atIndex:18];
        [enc setBuffer:bufStrides     offset:0 atIndex:19];
        [enc setBytes:&domain_size length:sizeof(domain_size) atIndex:20];
        [enc setBuffer:bufCustomData offset:0 atIndex:21];
        [enc setBuffer:bufCustomOff  offset:0 atIndex:22];
        [enc setBuffer:bufCustomNc   offset:0 atIndex:23];
        [enc setBytes:&custom_lo_bound length:sizeof(custom_lo_bound) atIndex:24];
        [enc setBuffer:bufX   offset:0 atIndex:25];
        [enc setBuffer:bufZi  offset:0 atIndex:26];
        [enc setBuffer:bufXis offset:0 atIndex:27];
        [enc setBytes:&dst_stride length:sizeof(dst_stride) atIndex:28];
        const NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreads:MTLSizeMake(n_threads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];

        // dest.inverse post-op. Chain a second encoder on the same command
        // buffer so the GPU sees the VM output already in-place on bufDst
        // — no host round-trip between kernels. gl3_inv_k can run in-place
        // safely because each thread only touches its own [tid*3..tid*3+2]
        // range and reads values into thread registers before writing back.
        if (dest_inverse && dest_dim == 3u) {
            id<MTLComputePipelineState> inv_pso = get_or_make_pso(ctx, @"gl3_inv_k");
            id<MTLComputeCommandEncoder> enc2 = [cmd computeCommandEncoder];
            [enc2 setComputePipelineState:inv_pso];
            [enc2 setBuffer:bufDst offset:bufDstOffset atIndex:0];  // input
            [enc2 setBuffer:bufDst offset:bufDstOffset atIndex:1];  // output (alias for in-place)
            [enc2 setBytes:&n_threads length:sizeof(n_threads) atIndex:2];
            const NSUInteger tg2 = std::min<NSUInteger>(inv_pso.maxTotalThreadsPerThreadgroup, 256);
            [enc2 dispatchThreads:MTLSizeMake(n_threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];
            [enc2 endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: expr_vm_min cmd error: %s\n", [msg UTF8String]);
            std::abort();
        }
        // Skip the copy-back when dst lives in a registered shared buffer
        // — the kernel wrote directly to the caller's memory via the
        // resolved offset, and Shared-mode Metal buffers are CPU-visible
        // after waitUntilCompleted.
        if (!dstResolvedShared) {
            if (dst_stride > dest_dim) {
                // Strided dest (e.g. imPol writing column `stagePos` of a
                // wider cm-section row). The kernel only wrote cells at
                // `tid*stride + [0, dest_dim)` into scratch; everything
                // else in scratch is uninitialised garbage. A dense
                // memcpy would stomp the neighbouring cells in `dst`
                // that the caller expects to preserve (other imPols in
                // the same cm-section row). Copy only the cells the
                // kernel actually wrote.
                const uint64_t* src = static_cast<const uint64_t*>([bufDst contents]);
                uint64_t*       d   = static_cast<uint64_t*>(dst);
                for (uint32_t t = 0; t < n_threads; ++t) {
                    const uint32_t base = t * dst_stride;
                    for (uint32_t c = 0; c < dest_dim; ++c) {
                        d[base + c] = src[base + c];
                    }
                }
            } else {
                std::memcpy(dst, [bufDst contents], dst_bytes);
            }
        }
    }
}

namespace {

// Helper: log2 of a power-of-two value. Aborts if not a power of two.
uint32_t log2_pow2_or_die(uint64_t x, const char* label) {
    if (x == 0 || (x & (x - 1)) != 0) {
        std::fprintf(stderr, "pil2::metal: %s=%llu is not a power of two\n",
                     label, (unsigned long long)x);
        std::abort();
    }
    return static_cast<uint32_t>(__builtin_ctzll(x));
}

} // namespace

namespace {

// Shared implementation behind ntt_forward_metal / ntt_inverse_metal. The two
// differ only in a final intt_reorder_scale dispatch (when inverse). `inv_n`
// is ignored unless inverse == true.
void ntt_dispatch(ContextHandle   ctx,
                  uint64_t*       data,
                  uint64_t        size,
                  uint64_t        ncols,
                  const uint64_t* roots,
                  uint64_t        roots_len,
                  bool            inverse,
                  uint64_t        inv_n) {
    if (!ctx)  { fatal("ntt_dispatch: null context"); }
    if (size == 0 || ncols == 0 || roots_len == 0) { return; }

    const uint32_t domain_pow  = log2_pow2_or_die(size,      "size");
    const uint32_t roots_pow   = log2_pow2_or_die(roots_len, "roots_len");
    if (roots_pow < domain_pow) {
        std::fprintf(stderr, "pil2::metal: roots_len=%llu < size=%llu\n",
                     (unsigned long long)roots_len, (unsigned long long)size);
        std::abort();
    }

    @autoreleasepool {
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];

        id<MTLComputePipelineState> pso_rev = get_or_make_pso(ctx, @"ntt_reverse_permutation");
        id<MTLComputePipelineState> pso_but = get_or_make_pso(ctx, @"ntt_butterfly_phase");
        id<MTLComputePipelineState> pso_r4  = get_or_make_pso(ctx, @"ntt_radix4_phase");
        id<MTLComputePipelineState> pso_r8  = get_or_make_pso(ctx, @"ntt_radix8_phase");
        id<MTLComputePipelineState> pso_f3  = get_or_make_pso(ctx, @"ntt_rev_butterfly_s1s2s3");
        id<MTLComputePipelineState> pso_rs  = get_or_make_pso(ctx, @"intt_reorder_scale");

        const size_t data_bytes  = static_cast<size_t>(size) * ncols * sizeof(uint64_t);
        const size_t roots_bytes = static_cast<size_t>(roots_len) * sizeof(uint64_t);

        // Pre-scratch-pool versions of this path did per-call
        // `newBufferWithBytes` for all three buffers. At data_bytes =
        // 2^23 × 3 × 8 = 192MB (computeQ) or 2^22 × 6 × 8 = 192MB
        // (computeLEv), that's hundreds of MB allocated + memcpy'd each
        // call. Reusing slot 0/1/2 of the context pool makes the second
        // and later calls effectively alloc-free.
        //
        // B.19 zero-copy: if `data` sits in a Metal-registered allocation
        // (typical case for aux_trace slices in computeQ / computeLEv),
        // bind that buffer directly — saves the IN and final-OUT memcpy.
        // bufSrc stays a separate scratch because the fused rev+s1s2s3
        // kernel reads src at bit-reversed positions while writing dst
        // at natural positions; sharing the same buffer would cause
        // cross-threadgroup read/write races.
        id<MTLBuffer> bufDst = nil;
        size_t bufDstOffset = 0;
        bool dataResolved = metal_resolve_shared(ctx, data, data_bytes, &bufDst, &bufDstOffset);
        if (!dataResolved) {
            bufDst = scratch_borrow(ctx, 0, data_bytes);
            std::memcpy([bufDst contents], data, data_bytes);
            bufDstOffset = 0;
        }
        id<MTLBuffer> bufRoots = scratch_borrow(ctx, 1, roots_bytes);
        std::memcpy([bufRoots contents], roots, roots_bytes);

        id<MTLBuffer> bufSrc = nil;
        uint32_t start_s = 1;
        if (domain_pow >= 3) {
            // Slot 2 is our secondary big-scratch for the fused rev+s1s2s3
            // kernel's read-only source. It holds the same initial data as
            // bufDst; the kernel reads bufSrc at bit-reversed indices and
            // writes bufDst in natural order.
            bufSrc = scratch_borrow(ctx, 2, data_bytes);
            std::memcpy([bufSrc contents], data, data_bytes);
            start_s = 4;  // fused kernel consumes stages 1, 2, 3
        }

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // Phase 0: reverse-permute + (maybe) first three butterfly stages.
        //
        // For domain_pow >= 3 we dispatch the fused rev+s1s2s3 kernel which
        // reads `src` at bit-reversed positions, applies stages 1/2/3 in
        // registers, and writes `dst` in natural order. Subsequent butterfly
        // phases run in-place on `dst` starting at s=4. This eliminates a
        // full domain-size memory round-trip vs. the "rev-perm pass, then
        // phase-1 pass" sequence used for smaller domains.
        //
        // For domain_pow < 3 (N < 8) the fused kernel isn't meaningful
        // (its 8-per-thread layout requires N >= 8), so we fall back to the
        // explicit rev-perm + phase-1 path. These small sizes are not on
        // the performance-critical path; the fall-back is for correctness.
        if (domain_pow >= 3) {
            // Twiddle constants for stages 2 and 3, derived from the roots
            // table: roots[k] = g^k where g is the primitive (roots_len)-th
            // root. So ω_4 = roots[roots_len/4], ω_8 = roots[roots_len/8],
            // ω_8^3 = roots[3*roots_len/8].
            const uint64_t I_val   = roots[roots_len >> 2];
            const uint64_t W8_val  = roots[roots_len >> 3];
            const uint64_t W8c_val = roots[(roots_len >> 3) * 3];

            [enc setComputePipelineState:pso_f3];
            [enc setBuffer:bufSrc offset:0 atIndex:0];
            [enc setBuffer:bufDst offset:bufDstOffset atIndex:1];
            uint32_t dp = domain_pow;
            uint32_t nc = static_cast<uint32_t>(ncols);
            [enc setBytes:&dp      length:sizeof(dp)      atIndex:2];
            [enc setBytes:&nc      length:sizeof(nc)      atIndex:3];
            [enc setBytes:&I_val   length:sizeof(I_val)   atIndex:4];
            [enc setBytes:&W8_val  length:sizeof(W8_val)  atIndex:5];
            [enc setBytes:&W8c_val length:sizeof(W8c_val) atIndex:6];

            const uint64_t grid = (size / 8) * ncols;
            const NSUInteger tg =
                std::min<NSUInteger>(pso_f3.maxTotalThreadsPerThreadgroup, 256);
            [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        } else {
            [enc setComputePipelineState:pso_rev];
            [enc setBuffer:bufDst offset:bufDstOffset atIndex:0];
            uint32_t dp = domain_pow;
            uint32_t nc = static_cast<uint32_t>(ncols);
            [enc setBytes:&dp length:sizeof(dp) atIndex:1];
            [enc setBytes:&nc length:sizeof(nc) atIndex:2];
            const NSUInteger tg = std::min<NSUInteger>(pso_rev.maxTotalThreadsPerThreadgroup, 256);
            [enc dispatchThreads:MTLSizeMake(size, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        // Phases: prefer radix-8 (3 stages) > radix-4 (2 stages) > radix-2
        // (1 stage), picking whichever fits the remaining stage count. This
        // minimises global-memory round-trips — 3 stages/dispatch for the
        // bulk when log2(N) is large, with smaller-radix tail kernels
        // covering any remainder. All three produce algebraically identical
        // results to back-to-back radix-2, so bit-exactness is preserved.
        //
        // `start_s` is 4 when the fused s1s2s3 kernel already consumed
        // stages 1..3, and 1 otherwise.
        {
            uint32_t nc   = static_cast<uint32_t>(ncols);
            uint32_t ds   = static_cast<uint32_t>(size);
            uint32_t s    = start_s;
            while (s <= domain_pow) {
                if (s + 2 <= domain_pow) {
                    // Radix-8: covers stages (s, s+1, s+2).
                    uint32_t stride_s = roots_pow - s;
                    [enc setComputePipelineState:pso_r8];
                    [enc setBuffer:bufDst   offset:bufDstOffset atIndex:0];
                    [enc setBuffer:bufRoots offset:0 atIndex:1];
                    [enc setBytes:&nc       length:sizeof(nc)       atIndex:2];
                    [enc setBytes:&ds       length:sizeof(ds)       atIndex:3];
                    [enc setBytes:&s        length:sizeof(s)        atIndex:4];
                    [enc setBytes:&stride_s length:sizeof(stride_s) atIndex:5];

                    const uint64_t grid = (size / 8) * ncols;
                    const NSUInteger tg = std::min<NSUInteger>(
                        pso_r8.maxTotalThreadsPerThreadgroup, 256);
                    [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    s += 3;
                } else if (s + 1 <= domain_pow) {
                    // Radix-4: covers stages (s, s+1).
                    uint32_t stride_s1 = roots_pow - (s + 1);
                    [enc setComputePipelineState:pso_r4];
                    [enc setBuffer:bufDst   offset:bufDstOffset atIndex:0];
                    [enc setBuffer:bufRoots offset:0 atIndex:1];
                    [enc setBytes:&nc        length:sizeof(nc)        atIndex:2];
                    [enc setBytes:&ds        length:sizeof(ds)        atIndex:3];
                    [enc setBytes:&s         length:sizeof(s)         atIndex:4];
                    [enc setBytes:&stride_s1 length:sizeof(stride_s1) atIndex:5];

                    const uint64_t grid = (size / 4) * ncols;
                    const NSUInteger tg = std::min<NSUInteger>(
                        pso_r4.maxTotalThreadsPerThreadgroup, 256);
                    [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    s += 2;
                } else {
                    // Radix-2 tail: single remaining stage.
                    uint32_t stride_shift = roots_pow - s;
                    [enc setComputePipelineState:pso_but];
                    [enc setBuffer:bufDst   offset:bufDstOffset atIndex:0];
                    [enc setBuffer:bufRoots offset:0 atIndex:1];
                    [enc setBytes:&nc           length:sizeof(nc)           atIndex:2];
                    [enc setBytes:&ds           length:sizeof(ds)           atIndex:3];
                    [enc setBytes:&s            length:sizeof(s)            atIndex:4];
                    [enc setBytes:&stride_shift length:sizeof(stride_shift) atIndex:5];

                    const uint64_t grid = (size / 2) * ncols;
                    const NSUInteger tg = std::min<NSUInteger>(
                        pso_but.maxTotalThreadsPerThreadgroup, 256);
                    [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    s += 1;
                }
            }
        }

        // INTT finalisation: (N-i) swap + 1/N scale, in-place on bufDst.
        // For forward NTT this is skipped; for inverse NTT this is the one
        // dispatch that converts the forward-shaped butterfly output into a
        // true inverse transform.
        if (inverse) {
            [enc setComputePipelineState:pso_rs];
            [enc setBuffer:bufDst offset:bufDstOffset atIndex:0];
            uint32_t ds = static_cast<uint32_t>(size);
            uint32_t nc = static_cast<uint32_t>(ncols);
            [enc setBytes:&ds    length:sizeof(ds)    atIndex:1];
            [enc setBytes:&nc    length:sizeof(nc)    atIndex:2];
            [enc setBytes:&inv_n length:sizeof(inv_n) atIndex:3];

            const uint64_t grid = (size_t)((size >> 1) + 1) * ncols;
            const NSUInteger tg = std::min<NSUInteger>(
                pso_rs.maxTotalThreadsPerThreadgroup, 256);
            [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: NTT cmd buffer error: %s\n", [msg UTF8String]);
            std::abort();
        }

        if (!dataResolved) {
            std::memcpy(data, [bufDst contents], data_bytes);
        }
        // When resolved, the kernel wrote straight into data's own
        // shared-memory backing store — Metal Shared storage is
        // CPU-visible post-waitUntilCompleted.
    }
}

} // namespace

void ntt_forward_metal(ContextHandle   ctx,
                       uint64_t*       data,
                       uint64_t        size,
                       uint64_t        ncols,
                       const uint64_t* roots,
                       uint64_t        roots_len) {
    ntt_dispatch(ctx, data, size, ncols, roots, roots_len, /*inverse=*/false, /*inv_n=*/0);
}

void ntt_inverse_metal(ContextHandle   ctx,
                       uint64_t*       data,
                       uint64_t        size,
                       uint64_t        ncols,
                       const uint64_t* roots,
                       uint64_t        roots_len,
                       uint64_t        inv_n) {
    ntt_dispatch(ctx, data, size, ncols, roots, roots_len, /*inverse=*/true, inv_n);
}

void lde_metal(ContextHandle   ctx,
               uint64_t*       output,
               const uint64_t* input,
               uint64_t        N_Extended,
               uint64_t        N,
               uint64_t        ncols,
               const uint64_t* roots,
               uint64_t        roots_len,
               const uint64_t* r_) {
    if (!ctx) { fatal("lde_metal: null context"); }
    if (N == 0 || N_Extended == 0 || ncols == 0 || roots_len == 0) { return; }

    const uint32_t dp_N   = log2_pow2_or_die(N,          "N");
    const uint32_t dp_Ext = log2_pow2_or_die(N_Extended, "N_Extended");
    const uint32_t dp_Rts = log2_pow2_or_die(roots_len,  "roots_len");
    if (dp_N > dp_Ext) {
        std::fprintf(stderr, "pil2::metal: lde_metal: N=%llu > N_Extended=%llu\n",
                     (unsigned long long)N, (unsigned long long)N_Extended);
        std::abort();
    }
    if (dp_Rts < dp_Ext) {
        std::fprintf(stderr, "pil2::metal: lde_metal: roots_len=%llu < N_Extended=%llu\n",
                     (unsigned long long)roots_len, (unsigned long long)N_Extended);
        std::abort();
    }

    @autoreleasepool {
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];

        id<MTLComputePipelineState> pso_rev = get_or_make_pso(ctx, @"ntt_reverse_permutation");
        id<MTLComputePipelineState> pso_but = get_or_make_pso(ctx, @"ntt_butterfly_phase");
        id<MTLComputePipelineState> pso_r4  = get_or_make_pso(ctx, @"ntt_radix4_phase");
        id<MTLComputePipelineState> pso_r8  = get_or_make_pso(ctx, @"ntt_radix8_phase");
        id<MTLComputePipelineState> pso_cs  = get_or_make_pso(ctx, @"intt_reorder_coset_scale");

        const size_t n_ext_bytes = static_cast<size_t>(N_Extended) * ncols * sizeof(uint64_t);
        const size_t n_bytes     = static_cast<size_t>(N)          * ncols * sizeof(uint64_t);
        const size_t roots_bytes = static_cast<size_t>(roots_len)        * sizeof(uint64_t);
        const size_t r_bytes     = static_cast<size_t>(N)                * sizeof(uint64_t);

        // Single N_Extended-sized buffer holding the input in its lower N rows
        // and zeros in the upper (N_Extended - N) rows. All three pipeline
        // stages (INTT butterflies over N, coset+scale over N, forward NTT
        // over N_Extended) operate in-place on this buffer — safe because
        // the first two stages only touch indices < N and leave the zero-
        // padded tail untouched.
        // Zero-copy path: if output is a registered shared buffer, use it
        // directly as bufExt (skips final N_Extended-sized memcpy back).
        id<MTLBuffer> bufExt     = nil;
        size_t        bufExtOff  = 0;
        bool          outResolved =
            metal_resolve_shared(ctx, output, n_ext_bytes, &bufExt, &bufExtOff);
        if (!outResolved) {
            bufExt    = scratch_borrow(ctx, 0, n_ext_bytes);
            bufExtOff = 0;
        }
        id<MTLBuffer> bufRoots = scratch_borrow(ctx, 1, roots_bytes);
        id<MTLBuffer> bufR     = scratch_borrow(ctx, 2, r_bytes);
        std::memcpy(static_cast<char*>([bufExt contents]) + bufExtOff, input, n_bytes);
        std::memset(static_cast<char*>([bufExt contents]) + bufExtOff + n_bytes, 0, n_ext_bytes - n_bytes);
        std::memcpy([bufRoots contents], roots, roots_bytes);
        std::memcpy([bufR contents],     r_,    r_bytes);

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        const uint32_t nc  = static_cast<uint32_t>(ncols);

        // -------- Step 1: INTT butterflies on bufExt[0..N] --------
        // Non-fused path: rev-perm then radix-8/4/2 phase loop. Not worth the
        // fused-rev-s1s2s3 optimisation here since the forward N_Extended NTT
        // dominates total time.
        {
            [enc setComputePipelineState:pso_rev];
            [enc setBuffer:bufExt offset:bufExtOff atIndex:0];
            uint32_t dp = dp_N;
            [enc setBytes:&dp length:sizeof(dp) atIndex:1];
            [enc setBytes:&nc length:sizeof(nc) atIndex:2];
            const NSUInteger tg =
                std::min<NSUInteger>(pso_rev.maxTotalThreadsPerThreadgroup, 256);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }
        {
            uint32_t ds = static_cast<uint32_t>(N);
            uint32_t s  = 1;
            while (s <= dp_N) {
                if (s + 2 <= dp_N) {
                    uint32_t stride_s = dp_Rts - s;
                    [enc setComputePipelineState:pso_r8];
                    [enc setBuffer:bufExt   offset:bufExtOff atIndex:0];
                    [enc setBuffer:bufRoots offset:0 atIndex:1];
                    [enc setBytes:&nc       length:sizeof(nc)       atIndex:2];
                    [enc setBytes:&ds       length:sizeof(ds)       atIndex:3];
                    [enc setBytes:&s        length:sizeof(s)        atIndex:4];
                    [enc setBytes:&stride_s length:sizeof(stride_s) atIndex:5];
                    const NSUInteger tg = std::min<NSUInteger>(
                        pso_r8.maxTotalThreadsPerThreadgroup, 256);
                    [enc dispatchThreads:MTLSizeMake((N / 8) * ncols, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    s += 3;
                } else if (s + 1 <= dp_N) {
                    uint32_t stride_s1 = dp_Rts - (s + 1);
                    [enc setComputePipelineState:pso_r4];
                    [enc setBuffer:bufExt   offset:bufExtOff atIndex:0];
                    [enc setBuffer:bufRoots offset:0 atIndex:1];
                    [enc setBytes:&nc        length:sizeof(nc)        atIndex:2];
                    [enc setBytes:&ds        length:sizeof(ds)        atIndex:3];
                    [enc setBytes:&s         length:sizeof(s)         atIndex:4];
                    [enc setBytes:&stride_s1 length:sizeof(stride_s1) atIndex:5];
                    const NSUInteger tg = std::min<NSUInteger>(
                        pso_r4.maxTotalThreadsPerThreadgroup, 256);
                    [enc dispatchThreads:MTLSizeMake((N / 4) * ncols, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    s += 2;
                } else {
                    uint32_t stride_shift = dp_Rts - s;
                    [enc setComputePipelineState:pso_but];
                    [enc setBuffer:bufExt   offset:bufExtOff atIndex:0];
                    [enc setBuffer:bufRoots offset:0 atIndex:1];
                    [enc setBytes:&nc           length:sizeof(nc)           atIndex:2];
                    [enc setBytes:&ds           length:sizeof(ds)           atIndex:3];
                    [enc setBytes:&s            length:sizeof(s)            atIndex:4];
                    [enc setBytes:&stride_shift length:sizeof(stride_shift) atIndex:5];
                    const NSUInteger tg = std::min<NSUInteger>(
                        pso_but.maxTotalThreadsPerThreadgroup, 256);
                    [enc dispatchThreads:MTLSizeMake((N / 2) * ncols, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    s += 1;
                }
            }
        }

        // -------- Step 2: intt_reorder_coset_scale on first N rows --------
        {
            [enc setComputePipelineState:pso_cs];
            [enc setBuffer:bufExt offset:bufExtOff atIndex:0];
            [enc setBuffer:bufR   offset:0 atIndex:1];
            uint32_t ds_N = static_cast<uint32_t>(N);
            [enc setBytes:&ds_N length:sizeof(ds_N) atIndex:2];
            [enc setBytes:&nc   length:sizeof(nc)   atIndex:3];
            const uint64_t grid = ((N >> 1) + 1) * ncols;
            const NSUInteger tg =
                std::min<NSUInteger>(pso_cs.maxTotalThreadsPerThreadgroup, 256);
            [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        // -------- Step 3: forward NTT on full bufExt (domain N_Extended) --------
        {
            [enc setComputePipelineState:pso_rev];
            [enc setBuffer:bufExt offset:bufExtOff atIndex:0];
            uint32_t dp = dp_Ext;
            [enc setBytes:&dp length:sizeof(dp) atIndex:1];
            [enc setBytes:&nc length:sizeof(nc) atIndex:2];
            const NSUInteger tg =
                std::min<NSUInteger>(pso_rev.maxTotalThreadsPerThreadgroup, 256);
            [enc dispatchThreads:MTLSizeMake(N_Extended, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }
        {
            uint32_t ds = static_cast<uint32_t>(N_Extended);
            uint32_t s  = 1;
            while (s <= dp_Ext) {
                if (s + 2 <= dp_Ext) {
                    uint32_t stride_s = dp_Rts - s;
                    [enc setComputePipelineState:pso_r8];
                    [enc setBuffer:bufExt   offset:bufExtOff atIndex:0];
                    [enc setBuffer:bufRoots offset:0 atIndex:1];
                    [enc setBytes:&nc       length:sizeof(nc)       atIndex:2];
                    [enc setBytes:&ds       length:sizeof(ds)       atIndex:3];
                    [enc setBytes:&s        length:sizeof(s)        atIndex:4];
                    [enc setBytes:&stride_s length:sizeof(stride_s) atIndex:5];
                    const NSUInteger tg = std::min<NSUInteger>(
                        pso_r8.maxTotalThreadsPerThreadgroup, 256);
                    [enc dispatchThreads:MTLSizeMake((N_Extended / 8) * ncols, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    s += 3;
                } else if (s + 1 <= dp_Ext) {
                    uint32_t stride_s1 = dp_Rts - (s + 1);
                    [enc setComputePipelineState:pso_r4];
                    [enc setBuffer:bufExt   offset:bufExtOff atIndex:0];
                    [enc setBuffer:bufRoots offset:0 atIndex:1];
                    [enc setBytes:&nc        length:sizeof(nc)        atIndex:2];
                    [enc setBytes:&ds        length:sizeof(ds)        atIndex:3];
                    [enc setBytes:&s         length:sizeof(s)         atIndex:4];
                    [enc setBytes:&stride_s1 length:sizeof(stride_s1) atIndex:5];
                    const NSUInteger tg = std::min<NSUInteger>(
                        pso_r4.maxTotalThreadsPerThreadgroup, 256);
                    [enc dispatchThreads:MTLSizeMake((N_Extended / 4) * ncols, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    s += 2;
                } else {
                    uint32_t stride_shift = dp_Rts - s;
                    [enc setComputePipelineState:pso_but];
                    [enc setBuffer:bufExt   offset:bufExtOff atIndex:0];
                    [enc setBuffer:bufRoots offset:0 atIndex:1];
                    [enc setBytes:&nc           length:sizeof(nc)           atIndex:2];
                    [enc setBytes:&ds           length:sizeof(ds)           atIndex:3];
                    [enc setBytes:&s            length:sizeof(s)            atIndex:4];
                    [enc setBytes:&stride_shift length:sizeof(stride_shift) atIndex:5];
                    const NSUInteger tg = std::min<NSUInteger>(
                        pso_but.maxTotalThreadsPerThreadgroup, 256);
                    [enc dispatchThreads:MTLSizeMake((N_Extended / 2) * ncols, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    s += 1;
                }
            }
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: LDE cmd buffer error: %s\n", [msg UTF8String]);
            std::abort();
        }

        if (!outResolved) {
            std::memcpy(output, [bufExt contents], n_ext_bytes);
        }
    }
}

namespace {

// Shared host dispatcher for poseidon2_permute_wN_batch. The three public
// width-specific entries below all funnel through here; they just pass a
// different W, kernel name, and C-array length.
void poseidon2_permute_generic(ContextHandle   ctx,
                                uint64_t*       out_states,
                                const uint64_t* in_states,
                                uint64_t        count,
                                const uint64_t* C,
                                const uint64_t* D,
                                uint32_t        W,
                                uint32_t        C_len,
                                NSString*       kernel_name) {
    if (!ctx) { fatal("poseidon2_permute: null context"); }
    if (count == 0) { return; }

    @autoreleasepool {
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];
        id<MTLComputePipelineState> pso = get_or_make_pso(ctx, kernel_name);

        const size_t state_bytes =
            static_cast<size_t>(count) * W * sizeof(uint64_t);
        const size_t C_bytes = static_cast<size_t>(C_len) * sizeof(uint64_t);
        const size_t D_bytes = static_cast<size_t>(W)     * sizeof(uint64_t);

        id<MTLBuffer> bufOut = scratch_borrow(ctx, 0, state_bytes);
        id<MTLBuffer> bufIn  = scratch_borrow(ctx, 1, state_bytes);
        id<MTLBuffer> bufC   = scratch_borrow(ctx, 2, C_bytes);
        id<MTLBuffer> bufD   = scratch_borrow(ctx, 3, D_bytes);
        std::memcpy([bufIn contents], in_states, state_bytes);
        std::memcpy([bufC  contents], C,         C_bytes);
        std::memcpy([bufD  contents], D,         D_bytes);

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];
        [enc setBuffer:bufIn  offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        [enc setBuffer:bufC   offset:0 atIndex:2];
        [enc setBuffer:bufD   offset:0 atIndex:3];
        uint32_t n = static_cast<uint32_t>(count);
        [enc setBytes:&n length:sizeof(n) atIndex:4];

        const NSUInteger tg =
            std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 64);
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: Poseidon2 cmd buffer error: %s\n",
                         [msg UTF8String]);
            std::abort();
        }

        std::memcpy(out_states, [bufOut contents], state_bytes);
    }
}

} // namespace

void poseidon2_permute_w8_metal(ContextHandle   ctx,
                                uint64_t*       out_states,
                                const uint64_t* in_states,
                                uint64_t        count,
                                const uint64_t* C,
                                const uint64_t* D) {
    poseidon2_permute_generic(ctx, out_states, in_states, count, C, D,
                              /*W=*/8, /*C_len=*/86,
                              @"poseidon2_permute_w8_batch");
}

void poseidon2_permute_w12_metal(ContextHandle   ctx,
                                 uint64_t*       out_states,
                                 const uint64_t* in_states,
                                 uint64_t        count,
                                 const uint64_t* C,
                                 const uint64_t* D) {
    poseidon2_permute_generic(ctx, out_states, in_states, count, C, D,
                              /*W=*/12, /*C_len=*/118,
                              @"poseidon2_permute_w12_batch");
}

void poseidon2_permute_w16_metal(ContextHandle   ctx,
                                 uint64_t*       out_states,
                                 const uint64_t* in_states,
                                 uint64_t        count,
                                 const uint64_t* C,
                                 const uint64_t* D) {
    poseidon2_permute_generic(ctx, out_states, in_states, count, C, D,
                              /*W=*/16, /*C_len=*/150,
                              @"poseidon2_permute_w16_batch");
}

namespace {

// Walk num_rows down by arity; return the level count if num_rows is an
// exact power of arity, else abort. Mirrors the "no padding needed"
// invariant the public Merkle entries require.
uint32_t levels_pow_arity_or_die(uint64_t num_rows, uint32_t arity) {
    if (num_rows == 0 || arity < 2) {
        std::fprintf(stderr, "pil2::metal: bad merkle params num_rows=%llu arity=%u\n",
                     (unsigned long long)num_rows, arity);
        std::abort();
    }
    uint32_t k = 0;
    while (num_rows > 1) {
        if (num_rows % arity != 0) {
            std::fprintf(stderr, "pil2::metal: num_rows=%llu is not a power of arity=%u\n",
                         (unsigned long long)num_rows, arity);
            std::abort();
        }
        num_rows /= arity;
        ++k;
    }
    return k;
}

// Shared host orchestration for every Poseidon2 Merkle width. Each level
// feeds the previous level's output directly into the compress kernel,
// which reads arity * CAPACITY = W u64s per thread (one parent) and writes
// CAPACITY u64s. Requires num_rows == arity^k.
void merkletree_poseidon2_generic(ContextHandle   ctx,
                                   uint64_t*       tree,
                                   const uint64_t* input,
                                   uint64_t        num_rows,
                                   const uint64_t* C, uint32_t C_len,
                                   const uint64_t* D,
                                   uint32_t        W,
                                   uint32_t        RATE,
                                   uint32_t        arity,
                                   NSString*       kernel_leaf,
                                   NSString*       kernel_compress) {
    if (!ctx) { fatal("merkletree_poseidon2: null context"); }
    if (num_rows == 0) { return; }
    const uint32_t levels = levels_pow_arity_or_die(num_rows, arity);

    constexpr uint32_t CAPACITY = 4;

    @autoreleasepool {
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];
        id<MTLComputePipelineState> pso_leaf = get_or_make_pso(ctx, kernel_leaf);
        id<MTLComputePipelineState> pso_comp = get_or_make_pso(ctx, kernel_compress);

        const size_t input_bytes = static_cast<size_t>(num_rows) * RATE * sizeof(uint64_t);

        // Total nodes in a k-level arity-N tree over num_rows leaves:
        //   sum_{i=0..k} arity^(k-i) = (arity*num_rows - 1) / (arity - 1)
        const uint64_t total_nodes =
            (static_cast<uint64_t>(arity) * num_rows - 1ull) / (arity - 1ull);
        const size_t tree_elems = static_cast<size_t>(total_nodes) * CAPACITY;
        const size_t tree_bytes = tree_elems * sizeof(uint64_t);
        const size_t C_bytes    = static_cast<size_t>(C_len) * sizeof(uint64_t);
        const size_t D_bytes    = static_cast<size_t>(W)     * sizeof(uint64_t);

        // B.19 zero-copy: tree → const_tree, input → aux_trace/const_pols.
        // Called once per commit stage (few times per proof), so resolver
        // cost amortises well against the multi-MB copies.
        id<MTLBuffer> bufTree = nil;
        size_t treeOffset = 0;
        bool treeResolved =
            metal_resolve_shared(ctx, tree, tree_bytes, &bufTree, &treeOffset);
        if (!treeResolved) {
            bufTree = scratch_borrow(ctx, 0, tree_bytes);
            treeOffset = 0;
        }
        id<MTLBuffer> bufInput = nil;
        size_t inputOffset = 0;
        bool inputResolved =
            metal_resolve_shared(ctx, input, input_bytes, &bufInput, &inputOffset);
        if (!inputResolved) {
            bufInput = scratch_borrow(ctx, 1, input_bytes);
            inputOffset = 0;
            std::memcpy([bufInput contents], input, input_bytes);
        }
        id<MTLBuffer> bufC     = scratch_borrow(ctx, 2, C_bytes);
        id<MTLBuffer> bufD     = scratch_borrow(ctx, 3, D_bytes);
        std::memcpy([bufC     contents], C,     C_bytes);
        std::memcpy([bufD     contents], D,     D_bytes);

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // Leaf layer.
        {
            [enc setComputePipelineState:pso_leaf];
            [enc setBuffer:bufInput offset:inputOffset atIndex:0];
            [enc setBuffer:bufTree  offset:treeOffset  atIndex:1];
            [enc setBuffer:bufC     offset:0 atIndex:2];
            [enc setBuffer:bufD     offset:0 atIndex:3];
            uint32_t n = static_cast<uint32_t>(num_rows);
            [enc setBytes:&n length:sizeof(n) atIndex:4];
            const NSUInteger tg = std::min<NSUInteger>(
                pso_leaf.maxTotalThreadsPerThreadgroup, 64);
            [enc dispatchThreads:MTLSizeMake(num_rows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        // Parent layers bottom-up. Each thread reads arity * CAPACITY = W
        // u64s starting at layer_base[l-1] + tid*W and writes CAPACITY
        // u64s to layer_base[l] + tid*CAPACITY.
        {
            uint64_t layer_base_elems = 0;
            uint64_t level_rows       = num_rows;
            for (uint32_t l = 1; l <= levels; ++l) {
                const uint64_t next_base  = layer_base_elems + level_rows * CAPACITY;
                const uint64_t next_count = level_rows / arity;

                [enc setComputePipelineState:pso_comp];
                [enc setBuffer:bufTree offset:treeOffset + layer_base_elems * sizeof(uint64_t) atIndex:0];
                [enc setBuffer:bufTree offset:treeOffset + next_base        * sizeof(uint64_t) atIndex:1];
                [enc setBuffer:bufC    offset:0 atIndex:2];
                [enc setBuffer:bufD    offset:0 atIndex:3];
                uint32_t n = static_cast<uint32_t>(next_count);
                [enc setBytes:&n length:sizeof(n) atIndex:4];
                const NSUInteger tg = std::min<NSUInteger>(
                    pso_comp.maxTotalThreadsPerThreadgroup, 64);
                [enc dispatchThreads:MTLSizeMake(next_count, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];

                layer_base_elems = next_base;
                level_rows       = next_count;
            }
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: Merkle cmd buffer error: %s\n",
                         [msg UTF8String]);
            std::abort();
        }

        if (!treeResolved) {
            std::memcpy(tree, [bufTree contents], tree_bytes);
        }
    }
}

} // namespace

void merkletree_poseidon2_w8_metal(ContextHandle   ctx,
                                   uint64_t*       tree,
                                   const uint64_t* input,
                                   uint64_t        num_rows,
                                   const uint64_t* C,
                                   const uint64_t* D) {
    merkletree_poseidon2_generic(ctx, tree, input, num_rows, C, /*C_len=*/86, D,
                                 /*W=*/8, /*RATE=*/4, /*arity=*/2,
                                 @"pose2_leaf_hash_w8", @"pose2_compress_w8");
}

void merkletree_poseidon2_w12_metal(ContextHandle   ctx,
                                    uint64_t*       tree,
                                    const uint64_t* input,
                                    uint64_t        num_rows,
                                    const uint64_t* C,
                                    const uint64_t* D) {
    merkletree_poseidon2_generic(ctx, tree, input, num_rows, C, /*C_len=*/118, D,
                                 /*W=*/12, /*RATE=*/8, /*arity=*/3,
                                 @"pose2_leaf_hash_w12", @"pose2_compress_w12");
}

void merkletree_poseidon2_w16_metal(ContextHandle   ctx,
                                    uint64_t*       tree,
                                    const uint64_t* input,
                                    uint64_t        num_rows,
                                    const uint64_t* C,
                                    const uint64_t* D) {
    merkletree_poseidon2_generic(ctx, tree, input, num_rows, C, /*C_len=*/150, D,
                                 /*W=*/16, /*RATE=*/12, /*arity=*/4,
                                 @"pose2_leaf_hash_w16", @"pose2_compress_w16");
}

// W=16 Merkle tree over leaves of `num_cols` u64s each (not just RATE). Leaf
// layer runs pose2_linear_hash_w16 (sponge absorb); parent layers reuse the
// existing pose2_compress_w16 kernel. Bit-exact with
// Poseidon2Goldilocks<16>::merkletree + linear_hash_seq when num_cols >= 1
// and num_rows == arity^k.
namespace {

// Shared multi-col Merkle dispatcher used by the three W={8,12,16} entries
// below. Leaf kernel is `kernel_leaf` (sponge-absorb variant); parent
// compress kernel is `kernel_compress`. All per-layer padding follows
// Poseidon2Goldilocks<W>::merkletree_seq exactly, so the output is
// bit-exact with the CPU reference.
void merkletree_poseidon2_cols_generic(ContextHandle   ctx,
                                       uint64_t*       tree,
                                       const uint64_t* input,
                                       uint64_t        num_cols,
                                       uint64_t        num_rows,
                                       const uint64_t* C, uint32_t C_len,
                                       const uint64_t* D,
                                       uint32_t        W,
                                       uint32_t        arity,
                                       NSString*       kernel_leaf,
                                       NSString*       kernel_compress,
                                       const char*     label) {
    if (!ctx) { std::fprintf(stderr, "%s: null ctx\n", label); std::abort(); }
    if (num_rows == 0) { return; }
    constexpr uint32_t CAPACITY = 4;

    // Total nodes including per-level padding, mirroring
    // MerkleTreeGL::getNumNodes exactly. For pow-of-arity num_rows this
    // simplifies to (arity*num_rows - 1) / (arity - 1); for non-pow-of-
    // arity it's larger by the cumulative extraZeros.
    uint64_t total_nodes = num_rows;
    {
        uint64_t level_n = num_rows;
        while (level_n > 1) {
            uint64_t extra = (arity - (level_n % arity)) % arity;
            total_nodes += extra;
            uint64_t next = (level_n + arity - 1) / arity;
            total_nodes += next;
            level_n = next;
        }
    }

    @autoreleasepool {
        id<MTLCommandQueue> q   = ctx->queues[current_stream_id()];
        id<MTLComputePipelineState> pso_leaf = get_or_make_pso(ctx, kernel_leaf);
        id<MTLComputePipelineState> pso_comp = get_or_make_pso(ctx, kernel_compress);

        const size_t input_bytes =
            static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols) * sizeof(uint64_t);
        const size_t tree_bytes = static_cast<size_t>(total_nodes) * CAPACITY * sizeof(uint64_t);
        const size_t C_bytes    = static_cast<size_t>(C_len) * sizeof(uint64_t);
        const size_t D_bytes    = static_cast<size_t>(W)     * sizeof(uint64_t);

        // B.19 zero-copy: tree → const_tree, input → aux_trace/const_pols.
        id<MTLBuffer> bufTree = nil;
        size_t treeOffset = 0;
        bool treeResolved =
            metal_resolve_shared(ctx, tree, tree_bytes, &bufTree, &treeOffset);
        if (!treeResolved) {
            bufTree = scratch_borrow(ctx, 0, tree_bytes);
            treeOffset = 0;
        }
        id<MTLBuffer> bufInput = nil;
        size_t inputOffset = 0;
        bool inputResolved =
            metal_resolve_shared(ctx, input, input_bytes, &bufInput, &inputOffset);
        if (!inputResolved) {
            bufInput = scratch_borrow(ctx, 1, input_bytes);
            inputOffset = 0;
            std::memcpy([bufInput contents], input, input_bytes);
        }
        id<MTLBuffer> bufC     = scratch_borrow(ctx, 2, C_bytes);
        id<MTLBuffer> bufD     = scratch_borrow(ctx, 3, D_bytes);
        std::memcpy([bufC     contents], C,     C_bytes);
        std::memcpy([bufD     contents], D,     D_bytes);

        id<MTLCommandBuffer>         cmd = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // Leaf layer — sponge absorb over `num_cols` elements per row.
        {
            [enc setComputePipelineState:pso_leaf];
            [enc setBuffer:bufInput offset:inputOffset atIndex:0];
            [enc setBuffer:bufTree  offset:treeOffset  atIndex:1];
            [enc setBuffer:bufC     offset:0 atIndex:2];
            [enc setBuffer:bufD     offset:0 atIndex:3];
            uint32_t n  = static_cast<uint32_t>(num_rows);
            uint32_t nc = static_cast<uint32_t>(num_cols);
            [enc setBytes:&n  length:sizeof(n)  atIndex:4];
            [enc setBytes:&nc length:sizeof(nc) atIndex:5];
            const NSUInteger tg = std::min<NSUInteger>(
                pso_leaf.maxTotalThreadsPerThreadgroup, 64);
            [enc dispatchThreads:MTLSizeMake(num_rows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }

        // Parent layers bottom-up. Mirrors Poseidon2Goldilocks<16>::merkletree_seq
        // exactly: at each level, zero-pad extraZeros rows so pending is a
        // multiple of arity, then compress nextN parents. The memset runs
        // on the host into the Metal shared-storage buffer; it writes to a
        // region no earlier dispatch touched and no later compress dispatch
        // reads before this encode, so the ordering inside a single command
        // buffer guarantees visibility at GPU execution time.
        {
            uint64_t layer_base_elems = 0;
            uint64_t pending          = num_rows;
            while (pending > 1) {
                const uint64_t extraZeros = (arity - (pending % arity)) % arity;
                if (extraZeros > 0) {
                    // Zero the padding slots: rows [pending, pending+extraZeros)
                    // in the current level, each CAPACITY u64s wide.
                    uint8_t* bytes_base = static_cast<uint8_t*>([bufTree contents]) + treeOffset;
                    uint64_t* layer_ptr = reinterpret_cast<uint64_t*>(bytes_base) + layer_base_elems;
                    std::memset(layer_ptr + pending * CAPACITY, 0,
                                extraZeros * CAPACITY * sizeof(uint64_t));
                }
                const uint64_t nextN     = (pending + arity - 1) / arity;
                const uint64_t next_base = layer_base_elems + (pending + extraZeros) * CAPACITY;

                [enc setComputePipelineState:pso_comp];
                [enc setBuffer:bufTree offset:treeOffset + layer_base_elems * sizeof(uint64_t) atIndex:0];
                [enc setBuffer:bufTree offset:treeOffset + next_base        * sizeof(uint64_t) atIndex:1];
                [enc setBuffer:bufC    offset:0 atIndex:2];
                [enc setBuffer:bufD    offset:0 atIndex:3];
                uint32_t n = static_cast<uint32_t>(nextN);
                [enc setBytes:&n length:sizeof(n) atIndex:4];
                const NSUInteger tg = std::min<NSUInteger>(
                    pso_comp.maxTotalThreadsPerThreadgroup, 64);
                [enc dispatchThreads:MTLSizeMake(nextN, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                layer_base_elems = next_base;
                pending          = nextN;
            }
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd error]) {
            NSString* msg = [[cmd error] localizedDescription];
            std::fprintf(stderr, "pil2::metal: %s cmd error: %s\n",
                         label, [msg UTF8String]);
            std::abort();
        }
        if (!treeResolved) {
            std::memcpy(tree, [bufTree contents], tree_bytes);
        }
    }
}

} // namespace (generic helper)

void merkletree_poseidon2_w8_cols_metal(ContextHandle   ctx,
                                        uint64_t*       tree,
                                        const uint64_t* input,
                                        uint64_t        num_cols,
                                        uint64_t        num_rows,
                                        const uint64_t* C,
                                        const uint64_t* D) {
    merkletree_poseidon2_cols_generic(ctx, tree, input, num_cols, num_rows,
                                      C, /*C_len=*/86, D,
                                      /*W=*/8, /*arity=*/2,
                                      @"pose2_linear_hash_w8", @"pose2_compress_w8",
                                      "merkletree_w8_cols");
}

void merkletree_poseidon2_w12_cols_metal(ContextHandle   ctx,
                                         uint64_t*       tree,
                                         const uint64_t* input,
                                         uint64_t        num_cols,
                                         uint64_t        num_rows,
                                         const uint64_t* C,
                                         const uint64_t* D) {
    merkletree_poseidon2_cols_generic(ctx, tree, input, num_cols, num_rows,
                                      C, /*C_len=*/118, D,
                                      /*W=*/12, /*arity=*/3,
                                      @"pose2_linear_hash_w12", @"pose2_compress_w12",
                                      "merkletree_w12_cols");
}

void merkletree_poseidon2_w16_cols_metal(ContextHandle   ctx,
                                         uint64_t*       tree,
                                         const uint64_t* input,
                                         uint64_t        num_cols,
                                         uint64_t        num_rows,
                                         const uint64_t* C,
                                         const uint64_t* D) {
    merkletree_poseidon2_cols_generic(ctx, tree, input, num_cols, num_rows,
                                      C, /*C_len=*/150, D,
                                      /*W=*/16, /*arity=*/4,
                                      @"pose2_linear_hash_w16", @"pose2_compress_w16",
                                      "merkletree_w16_cols");
}

} // namespace pil2::metal

// ===========================================================================
// C API (metal_c_api.h) — thin extern "C" wrappers around pil2::metal::*
// for Rust / pure-C / future starks_metal.mm callers.
// ===========================================================================

#include "metal_c_api.h"

extern "C" int pil2_metal_available(void) {
    // The singleton init aborts if no Metal device is found, which would be
    // worse than returning 0 from a probe. Guard with a try/catch-style
    // check: attempt get_context() inside @try — on failure return 0.
    @try {
        auto* ctx = pil2::metal::get_context();
        return ctx != nullptr ? 1 : 0;
    } @catch (NSException* ex) {
        (void)ex;
        return 0;
    }
}

extern "C" size_t pil2_metal_device_name(char* out, size_t out_len) {
    if (!out || out_len == 0) return 0;
    auto* ctx = pil2::metal::get_context();
    std::string name = pil2::metal::device_name(ctx);
    size_t to_copy = std::min(name.size(), out_len - 1);
    std::memcpy(out, name.data(), to_copy);
    out[to_copy] = '\0';
    return to_copy;
}

extern "C" int pil2_metal_ntt_forward(uint64_t* data, uint64_t size, uint64_t ncols,
                                      const uint64_t* roots, uint64_t roots_len) {
    pil2::metal::ntt_forward_metal(pil2::metal::get_context(),
                                   data, size, ncols, roots, roots_len);
    return 0;
}

extern "C" int pil2_metal_ntt_inverse(uint64_t* data, uint64_t size, uint64_t ncols,
                                      const uint64_t* roots, uint64_t roots_len,
                                      uint64_t inv_n) {
    pil2::metal::ntt_inverse_metal(pil2::metal::get_context(),
                                   data, size, ncols, roots, roots_len, inv_n);
    return 0;
}

extern "C" int pil2_metal_lde(uint64_t* output, const uint64_t* input,
                              uint64_t N_Extended, uint64_t N, uint64_t ncols,
                              const uint64_t* roots, uint64_t roots_len,
                              const uint64_t* r_) {
    pil2::metal::lde_metal(pil2::metal::get_context(),
                           output, input, N_Extended, N, ncols,
                           roots, roots_len, r_);
    return 0;
}

extern "C" int pil2_metal_poseidon2_permute_w8(uint64_t* out_states, const uint64_t* in_states,
                                                uint64_t count, const uint64_t* C, const uint64_t* D) {
    pil2::metal::poseidon2_permute_w8_metal(pil2::metal::get_context(),
                                             out_states, in_states, count, C, D);
    return 0;
}

extern "C" int pil2_metal_poseidon2_permute_w12(uint64_t* out_states, const uint64_t* in_states,
                                                 uint64_t count, const uint64_t* C, const uint64_t* D) {
    pil2::metal::poseidon2_permute_w12_metal(pil2::metal::get_context(),
                                              out_states, in_states, count, C, D);
    return 0;
}

extern "C" int pil2_metal_poseidon2_permute_w16(uint64_t* out_states, const uint64_t* in_states,
                                                 uint64_t count, const uint64_t* C, const uint64_t* D) {
    pil2::metal::poseidon2_permute_w16_metal(pil2::metal::get_context(),
                                              out_states, in_states, count, C, D);
    return 0;
}

extern "C" int pil2_metal_merkletree_w8(uint64_t* tree, const uint64_t* input, uint64_t num_rows,
                                         const uint64_t* C, const uint64_t* D) {
    pil2::metal::merkletree_poseidon2_w8_metal(pil2::metal::get_context(),
                                                tree, input, num_rows, C, D);
    return 0;
}

extern "C" int pil2_metal_merkletree_w12(uint64_t* tree, const uint64_t* input, uint64_t num_rows,
                                          const uint64_t* C, const uint64_t* D) {
    pil2::metal::merkletree_poseidon2_w12_metal(pil2::metal::get_context(),
                                                 tree, input, num_rows, C, D);
    return 0;
}

extern "C" int pil2_metal_merkletree_w16(uint64_t* tree, const uint64_t* input, uint64_t num_rows,
                                          const uint64_t* C, const uint64_t* D) {
    pil2::metal::merkletree_poseidon2_w16_metal(pil2::metal::get_context(),
                                                 tree, input, num_rows, C, D);
    return 0;
}

extern "C" int pil2_metal_merkletree_w16_cols(uint64_t* tree, const uint64_t* input,
                                              uint64_t num_cols, uint64_t num_rows,
                                              const uint64_t* C, const uint64_t* D) {
    pil2::metal::merkletree_poseidon2_w16_cols_metal(pil2::metal::get_context(),
                                                     tree, input, num_cols, num_rows, C, D);
    return 0;
}

extern "C" int pil2_metal_merkletree_w8_cols(uint64_t* tree, const uint64_t* input,
                                             uint64_t num_cols, uint64_t num_rows,
                                             const uint64_t* C, const uint64_t* D) {
    pil2::metal::merkletree_poseidon2_w8_cols_metal(pil2::metal::get_context(),
                                                    tree, input, num_cols, num_rows, C, D);
    return 0;
}

extern "C" int pil2_metal_merkletree_w12_cols(uint64_t* tree, const uint64_t* input,
                                              uint64_t num_cols, uint64_t num_rows,
                                              const uint64_t* C, const uint64_t* D) {
    pil2::metal::merkletree_poseidon2_w12_cols_metal(pil2::metal::get_context(),
                                                     tree, input, num_cols, num_rows, C, D);
    return 0;
}

#else  // !PIL2_HAS_METAL — C API stubs that return unavailable

#include "metal_c_api.h"
#include <cstring>

extern "C" int pil2_metal_available(void) { return 0; }
extern "C" size_t pil2_metal_device_name(char* out, size_t out_len) {
    if (out && out_len > 0) out[0] = '\0';
    return 0;
}
extern "C" int pil2_metal_ntt_forward(uint64_t*, uint64_t, uint64_t, const uint64_t*, uint64_t) { return -1; }
extern "C" int pil2_metal_ntt_inverse(uint64_t*, uint64_t, uint64_t, const uint64_t*, uint64_t, uint64_t) { return -1; }
extern "C" int pil2_metal_lde(uint64_t*, const uint64_t*, uint64_t, uint64_t, uint64_t, const uint64_t*, uint64_t, const uint64_t*) { return -1; }
extern "C" int pil2_metal_poseidon2_permute_w8 (uint64_t*, const uint64_t*, uint64_t, const uint64_t*, const uint64_t*) { return -1; }
extern "C" int pil2_metal_poseidon2_permute_w12(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, const uint64_t*) { return -1; }
extern "C" int pil2_metal_poseidon2_permute_w16(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, const uint64_t*) { return -1; }
extern "C" int pil2_metal_merkletree_w8 (uint64_t*, const uint64_t*, uint64_t, const uint64_t*, const uint64_t*) { return -1; }
extern "C" int pil2_metal_merkletree_w12(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, const uint64_t*) { return -1; }
extern "C" int pil2_metal_merkletree_w16(uint64_t*, const uint64_t*, uint64_t, const uint64_t*, const uint64_t*) { return -1; }
extern "C" int pil2_metal_merkletree_w16_cols(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*, const uint64_t*) { return -1; }
extern "C" int pil2_metal_merkletree_w8_cols (uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*, const uint64_t*) { return -1; }
extern "C" int pil2_metal_merkletree_w12_cols(uint64_t*, const uint64_t*, uint64_t, uint64_t, const uint64_t*, const uint64_t*) { return -1; }

#endif // PIL2_HAS_METAL
