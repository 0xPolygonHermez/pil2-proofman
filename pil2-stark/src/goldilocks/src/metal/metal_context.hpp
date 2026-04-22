#ifndef PIL2_METAL_CONTEXT_HPP
#define PIL2_METAL_CONTEXT_HPP

#include "../platform.hpp"

#if PIL2_HAS_METAL

#include <cstdint>
#include <string>

namespace pil2::metal {

// Opaque handle. The actual Context struct holds MTLDevice + MTLCommandQueue
// Obj-C pointers and is only defined inside metal_context.mm so pure-C++
// translation units never pull in <Metal/Metal.h>.
struct Context;
using ContextHandle = Context*;

// Returns the process-wide Metal context singleton. First call creates the
// MTLDevice and MTLCommandQueue; subsequent calls return the cached pointer.
// Thread-safe via std::call_once. Aborts with a diagnostic if no Metal
// device is available (macOS on any supported model always has one, so this
// only fires in sandboxed / headless CI without Metal — we want loud, not
// silent, failure).
ContextHandle get_context();

// Unified-memory allocator (B.18). Allocates a Metal-backed shared-storage
// buffer of the requested size, registers it in the process context, and
// returns the CPU-accessible `.contents` pointer. The caller may read and
// write the pointer directly; Metal kernels can bind the same buffer
// zero-copy (the VM bridge and other kernels lookup ptr → MTLBuffer
// internally). Caller must eventually release with metal_free_shared().
// Returns nullptr on bytes == 0. Aborts on allocation failure.
void* metal_alloc_shared(uint64_t bytes);

// Release a buffer previously returned by metal_alloc_shared. Pointer must
// match exactly what alloc returned (not a derived pointer inside the
// allocation). Silent on unknown pointer (logs to stderr).
void metal_free_shared(void* ptr);

// True when `ptr` is exactly the base of a registered allocation.
bool metal_is_shared_base(const void* ptr);

// Human-readable device name (e.g. "Apple M4 Pro"). Diagnostics only.
std::string device_name(ContextHandle ctx);

// Phase C3 smoke: compiles an embedded MSL "noop" kernel, dispatches it on a
// single thread to write 42 into a shared buffer, waits for completion, and
// returns the value read back. Exists so the test harness can validate the
// full host→GPU pipeline (source compile, pipeline state, command buffer,
// waitUntilCompleted) through a single C++ entry point — without .mm test
// code or raw Obj-C handles leaking into pure-C++ translation units.
//
// Aborts loudly with a diagnostic on any Metal API failure (library compile
// error, PSO creation failure, command buffer error). That's fine for a
// smoke harness: if Metal is broken, we want a visible stack, not silent 0.
uint32_t run_noop_test(ContextHandle ctx);

// Phase C4: applies one of the Goldilocks field ops to two u64 arrays
// pairwise and writes the canonical [0, p) result into `out`. `op` is one
// of "add", "sub", "mul".
//
// Inputs must satisfy a[i], b[i] in [0, p). Output is fully reduced.
// Compiles the MSL on every call (Metal's shader cache makes repeats
// nearly free). `n == 0` is a no-op.
//
// Exists so the correctness test in tests/test_metal_field.cpp can be
// pure C++ (no Obj-C types leak) and still exercise the kernel path end-
// to-end. The public NTT / Merkle kernels that come in C5-C8 will not go
// through this helper — they'll get dedicated entry points.
void run_field_op(ContextHandle   ctx,
                  const char*     op,
                  const uint64_t* a,
                  const uint64_t* b,
                  uint64_t*       out,
                  uint32_t        n);

// FRI fold (cubic extension), specialised to nX = 8 (prev_bits - current_bits
// == 3). Bit-exact with FRI<Goldilocks::Element>::fold for the step != 0
// branch. `pol` is in/out, sized `nX * pol2N * FIELD_EXTENSION` u64 on
// input and `pol2N * FIELD_EXTENSION` on output (the kernel overwrites
// the first `pol2N * FIELD_EXTENSION` entries in place and leaves the
// rest untouched — caller is responsible for the trim).
//
// polShiftInv, wi, inv8 and the 8-th roots table are computed by the
// caller (the bridge at starkpil/fri/.. knows the values from
// nBitsExt / prevBits / currentBits and `step`). challenge is 3 u64.
void fri_fold_w8_metal(ContextHandle   ctx,
                       uint64_t*       pol,
                       const uint64_t* challenge,
                       uint64_t        pol2N,
                       uint64_t        polShiftInv,
                       uint64_t        wi,
                       uint64_t        inv8,
                       const uint64_t* roots8);

// Cubic-extension test helper. op ∈ {"add","sub","mul","mul_scalar"}.
// - add/sub/mul: a, b, out are each `n * 3` u64.
// - mul_scalar : a, out are `n * 3` u64; b is `n` u64 (one base-field
//                scalar per gl3 element).
// Used by tests/test_metal_gl3.cpp; also available to callers needing a
// black-box kernel path to validate the gl3_* device functions.
void run_gl3_op(ContextHandle   ctx,
                const char*     op,
                const uint64_t* a,
                const uint64_t* b,
                uint64_t*       out,
                uint32_t        n);

// Production gl3 element-wise multiply (cubic × cubic → cubic), built on
// gl3_mul_k (dense dst) or gl3_mul_strided_k (dst_stride > 3). PSO is
// cached and a/b/out go through metal_resolve_shared, so pointers from
// metal_alloc_shared bind zero-copy; heap pointers fall through the
// persistent scratch pool. Dispatches on the per-thread stream queue.
//
// Layout: a, b are each n * 3 u64 (dense). Dst receives 3 u64s per thread
// at `out[tid * dst_stride + c]` for c in [0, 3). dst_stride defaults to 3
// (dense); values >3 leave the cells in (3, dst_stride) untouched, so the
// caller can write one cubic into a wider cm-section row alongside other
// imPols without disturbing them.
void run_gl3_mul(ContextHandle   ctx,
                 const uint64_t* a,
                 const uint64_t* b,
                 uint64_t*       out,
                 uint32_t        n,
                 uint32_t        dst_stride = 3);

// Isolated unit-test runners for the Fermat / cubic inverses used by the
// expression VM's xi source (B.11). Each applies the inverse element-wise
// and returns canonical results. Caller must ensure no element is zero.
void run_gl_inv_test(ContextHandle   ctx,
                     const uint64_t* a,
                     uint64_t*       out,
                     uint32_t        n);

void run_gl3_inv_test(ContextHandle   ctx,
                      const uint64_t* a,
                      uint64_t*       out,
                      uint32_t        n);

// STEP_EVALS evmap (B.16). Computes, per eval e in [0, n_evals):
//   evals_out[e] = sum_{k=0..N-1} LEv[opening_pos[e], k] ·
//                                  pol_at_row(k << extend_bits)
// Input layout:
//   lev         : N * np * 3 u64 (row-major by k, then opening, then c).
//   aux_trace / custom / const_pols : three candidate backing buffers;
//                                     per-eval `buf_ids[e]` picks one
//                                     (0, 1, 2 respectively). Any of
//                                     the three may be null when unused.
//   offsets[e]  : u64 element-index into the chosen buffer.
//   strides[e]  : u64 row stride in the chosen buffer.
//   dims[e]     : 1 (base field, widens to [v,0,0]) or 3 (cubic).
//   opening_pos[e] : index into LEv's np axis.
// Output: evals_out is n_evals * 3 u64, canonical [0, p).
// One Metal threadgroup handles one eval; threads sum a strided
// partition of rows then tree-reduce in shared memory.
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
                     uint32_t         const_pols_len_u64);

// Op-dispatcher test helper. `flavor` ∈ {"gl_op","gl3_op","gl3_op_31"}:
//   gl_op      — a, b, out each n u64 (base×base → base)
//   gl3_op     — a, b, out each n*3 u64 (cubic×cubic → cubic)
//   gl3_op_31  — a, out each n*3 u64; b is n u64 (cubic×base → cubic,
//                op ∈ {0,1,2,3} with op_31 semantics)
// `op_code` ∈ {0,1,2,3} = {add, sub, mul, reverse-sub}.
void run_op_dispatch(ContextHandle   ctx,
                     const char*     flavor,
                     uint32_t        op_code,
                     const uint64_t* a,
                     const uint64_t* b,
                     uint64_t*       out,
                     uint32_t        n);

// Expression-VM minimum-viable kernel (Phase E, Steps B.0..B.5). Runs a
// tiny straight-line bytecode program once per thread (tid == row). The
// kernel supports only outer-op 0 (dim1×dim1→dim1). Operand source types
// covered so far (all base-field / dim 1):
//   - type == 0                      : const_pols. Read as const_pols
//                                      [stage_offsets[0] + tid *
//                                      stage_ncols[0] + slot]. Caller
//                                      supplies the extended-tree address
//                                      when domain_extended, else the raw
//                                      const-pols address.
//   - type == 1 && !domain_extended  : trace (stage=1 raw buffer).
//                                      Read as trace[tid * stage_ncols[1]
//                                      + slot].
//   - type == 1 &&  domain_extended  : aux_trace at stage_offsets[1]
//                                      (stage 1 lives inside aux_trace on
//                                      the extended domain).
//   - type in [2, nStages+1]         : aux_trace for that stage.
//                                      Read as aux_trace[stage_offsets[type]
//                                      + tid * stage_ncols[type] + slot].
//   - type == buffer_commits_size    : tmp1 per-thread scratch.
//   - type == buffer_commits_size+2  : public_inputs[slot]   (flat)
//   - type == buffer_commits_size+3  : numbers[slot]         (flat)
//   - type == buffer_commits_size+4  : air_values[slot]      (flat)
//   - type == buffer_commits_size+5  : proof_values[slot]    (flat)
//   - type == buffer_commits_size+6  : airgroup_values[slot] (flat)
//   - type == buffer_commits_size+7  : challenges[slot]      (flat)
//   - type == buffer_commits_size+8  : evals[slot]           (flat)
//
// "Flat" sources are read as a single scalar indexed by slot (no per-row
// stride, no base offset).
//
//   ops           — length n_ops u8, every entry MUST be 0.
//   args          — length n_args u16; consumed 8 entries per op (standard
//                   VM layout: inner_op, dest_slot, typeA, slotA,
//                   rowOffsetA, typeB, slotB, rowOffsetB — rowOffset
//                   entries unused until cyclic / nonzero-offset lands).
//   numbers       — length n_numbers u64 (may be 0 / nullptr).
//   trace         — length trace_len_u64 u64 (nullable when no op
//                   references type 1; caller must set trace_len_u64 = 0
//                   in that case).
//   aux_trace     — length aux_trace_len_u64 u64 (nullable when no op
//                   references types in [2, nStages+1]).
//   stage_offsets — length n_stages_plus_2 u32; stage_offsets[t] is the
//                   aux_trace base offset for type t (u64 units). Must be
//                   non-null; pass a single-element dummy if no aux_trace
//                   load is used.
//   stage_ncols   — length n_stages_plus_2 u32; stage_ncols[t] is the row
//                   stride in u64 units for type t (trace uses index 1,
//                   aux_trace uses index t).
//   dst           — length n_threads u64; final op of every thread writes
//                   here.
//   buffer_commits_size — must match the sentinel the bytecode was
//                         compiled against (= setup's bufferCommitsSize).
//
// tmp1[] is thread-local with a compile-time max of 64 slots. Callers must
// not use tmp-slot indices >= 64.
// Prover-side helpers for type == nStages+2 (aka customCommitsLoBound - 2)
// and type == nStages+3 (aka customCommitsLoBound - 1, the xi source).
// Bytecode's `slot` field:
//   - proverHelpers: slot == 0 reads x_current[row] (caller supplies x
//     when domain_extended, else x_n). slot >= 1 reads
//     zi[(slot - 1) * domain_size + row].
//   - xi (cubic, compute-on-read): slot is the opening-point index o.
//     Kernel computes cubic (x_current[row] - xis[o]) and returns its
//     cubic inverse. xis is n_opening_points * 3 u64, one cubic per
//     opening point.
// All three are nullable: pass nullptr + length 0 when no op references
// the corresponding source.
struct ExprVmProverHelpers {
    const uint64_t* x_current         = nullptr;
    const uint64_t* zi                = nullptr;
    const uint64_t* xis               = nullptr;
    uint32_t        x_current_len_u64 = 0;
    uint32_t        zi_len_u64        = 0;
    uint32_t        xis_len_u64       = 0;
};

// Custom-commit fixed columns (ROM-like per-AIR data). Each commit has
// its own offset within the shared `data` buffer and its own per-row
// column count; bytecode types in [bCS - count, bCS) index into these
// tables. Zero custom commits (count == 0, all nullptr) is the default.
struct ExprVmCustomCommits {
    const uint64_t* data         = nullptr;
    const uint32_t* offsets      = nullptr;   // length `count`
    const uint32_t* ncols        = nullptr;   // length `count`
    uint32_t        data_len_u64 = 0;
    uint32_t        count        = 0;
};

// Flat scalar tables addressed only by slot (no per-row stride). Each
// field is optional: nullptr + length 0 means "no bytecode op references
// this source". Added incrementally (B.5); future sources can extend this
// struct without churning call sites.
struct ExprVmFlatTables {
    const uint64_t* public_inputs          = nullptr;
    const uint64_t* air_values             = nullptr;
    const uint64_t* proof_values           = nullptr;
    const uint64_t* airgroup_values        = nullptr;
    const uint64_t* challenges             = nullptr;
    const uint64_t* evals                  = nullptr;
    uint32_t        public_inputs_len_u64   = 0;
    uint32_t        air_values_len_u64      = 0;
    uint32_t        proof_values_len_u64    = 0;
    uint32_t        airgroup_values_len_u64 = 0;
    uint32_t        challenges_len_u64      = 0;
    uint32_t        evals_len_u64           = 0;
};

// next_strides indexes by the rowOffsetIdx field in bytecode (args[4] or
// args[7]). Each entry is a signed row delta in u64 row units (not bytes);
// e.g., {0, 1, -1} means offset-idx 0 reads current row, idx 1 reads
// next row, idx 2 reads previous row. The kernel wraps via pow-of-2
// bitmask, so domain_size must be a power of two.
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
                     const ExprVmFlatTables& flat = {},
                     uint32_t        dest_dim = 1,
                     const ExprVmCustomCommits& custom = {},
                     const ExprVmProverHelpers& prover_helpers = {},
                     bool            dest_inverse = false,
                     uint32_t        dst_stride = 0);

// Phase C5a: forward Cooley-Tukey NTT (decimation-in-time), minimum path.
// In-place on `data`. `size` must be a power of 2. `ncols` is the row-major
// stride per domain index (same contract as NTT_Goldilocks).
//
// `roots` holds the full primitive (roots_len)-th roots-of-unity table:
// roots[k] = g^k mod p, where g = Goldilocks::w(log2(roots_len)). This must
// be a power-of-2 length ≥ size so the per-phase stride (log2(roots_len) - s)
// is non-negative. The caller owns the table; it's copied into a Metal
// buffer on each call (cache amortisation is a later concern).
//
// Algorithm: bit-reverse permute `data`, then log2(size) radix-2 butterfly
// phases. Output is in natural order, bit-exact with NTT_Goldilocks::NTT.
//
// Scope note: this is the minimal C5a path. C5b adds radix-4/8 phases and
// fused rev+butterfly kernels for perf; C5c adds INTT + coset scale.
void ntt_forward_metal(ContextHandle   ctx,
                       uint64_t*       data,
                       uint64_t        size,
                       uint64_t        ncols,
                       const uint64_t* roots,
                       uint64_t        roots_len);

// Phase C5c: inverse NTT. Reuses the forward butterfly kernels — the math
// NTT(NTT(x))[n] = N * x[(N-n) mod N] means INTT(X) = (1/N) * reorder(NTT(X)),
// so we run the same forward path on `data` and then apply one fused
// intt_reorder_scale kernel at the end. `inv_n` must equal inv(size) in the
// Goldilocks field (the caller has Goldilocks::inv handy).
//
// Output is in natural order, bit-exact with NTT_Goldilocks::INTT.
void ntt_inverse_metal(ContextHandle   ctx,
                       uint64_t*       data,
                       uint64_t        size,
                       uint64_t        ncols,
                       const uint64_t* roots,
                       uint64_t        roots_len,
                       uint64_t        inv_n);

// Phase C6: Low-Degree Extension. Given `N` evaluations of a polynomial on
// the N-th roots of unity, produce `N_Extended` evaluations on the coset
// `shift * H_N_Extended`. Mirrors NTT_Goldilocks::LDE.
//
// Steps (all on GPU):
//   1. Copy input into the first N*ncols of a fresh N_Extended*ncols buffer,
//      zero-fill the rest.
//   2. INTT butterflies on the first N positions (forward butterfly kernels;
//      they don't touch indices >= N).
//   3. intt_reorder_coset_scale using the precomputed `r_` array — combines
//      the INTT reorder, 1/N scaling, and per-element coset shift into one
//      pass.
//   4. Forward NTT on the full N_Extended buffer.
//
// Caller contract:
//   - `N`, `N_Extended` are powers of two; N_Extended >= N.
//   - `roots` holds the primitive (roots_len)-th roots of unity;
//     roots_len >= N_Extended so the same buffer serves both transforms via
//     the existing per-phase stride_shift trick.
//   - `r_` has length N and equals r_[i] = shift^i / N mod p. Trivially,
//     r_[0] == inv(N).
void lde_metal(ContextHandle   ctx,
               uint64_t*       output,
               const uint64_t* input,
               uint64_t        N_Extended,
               uint64_t        N,
               uint64_t        ncols,
               const uint64_t* roots,
               uint64_t        roots_len,
               const uint64_t* r_);

// Phase C7: Poseidon2 permutation over W=8 states. Bit-exact with
// Poseidon2Goldilocks<8>::permute_seq.
//
// Applies the permutation to `count` independent 8-element states stored
// contiguously in `in_states` (length count*8), writes results into
// `out_states` (length count*8). Out-of-place; in-place aliasing is
// supported (the kernel reads into thread-local registers before writing).
//
// `C` is the 86-element round constants table (4 initial full rounds × 8 +
// 22 partial rounds × 1 + 4 final full rounds × 8). `D` is the 8-element
// internal-round diagonal. Both come from pil2-stark's
// Poseidon2GoldilocksConstants; the caller converts Goldilocks::Element to
// u64 before calling.
void poseidon2_permute_w8_metal(ContextHandle   ctx,
                                uint64_t*       out_states,
                                const uint64_t* in_states,
                                uint64_t        count,
                                const uint64_t* C,    // length 86
                                const uint64_t* D);   // length 8

// Poseidon2 permutation for W=12. C has length 118, D has length 12.
// Same contract as the W=8 variant. Used for arity-3 Merkle trees
// (3 × CAPACITY=4 inputs per compress fits the W=12 sponge state).
void poseidon2_permute_w12_metal(ContextHandle   ctx,
                                 uint64_t*       out_states,
                                 const uint64_t* in_states,
                                 uint64_t        count,
                                 const uint64_t* C,   // length 118
                                 const uint64_t* D);  // length 12

// Poseidon2 permutation for W=16. C has length 150, D has length 16.
// Used for arity-4 Merkle trees (4 × CAPACITY=4 = 16 = W fits directly).
void poseidon2_permute_w16_metal(ContextHandle   ctx,
                                 uint64_t*       out_states,
                                 const uint64_t* in_states,
                                 uint64_t        count,
                                 const uint64_t* C,   // length 150
                                 const uint64_t* D);  // length 16

// Phase C8: Poseidon2 W=8 Merkle tree (arity=2, RATE=4, CAPACITY=4).
//
// `num_rows` must be a power of two (enforces binary arity-2 reduction
// without extra-zero padding at each level). `input` holds `num_rows * RATE`
// u64s — one RATE=4 input block per leaf, corresponding to the scalar
// `linear_hash(..., size=RATE)` fast path.
//
// `tree` is filled with the flat `(2*num_rows - 1) * CAPACITY` tree layout
// used by Poseidon2Goldilocks<8>::merkletree_seq: leaves first, then
// level 1 parents, level 2 grandparents, etc. The last CAPACITY u64s are
// the Merkle root.
//
// Bit-exact with Poseidon2Goldilocks<8>::merkletree_seq(tree, input,
// num_cols=RATE, num_rows, arity=2, num_threads=*, dim=1).
void merkletree_poseidon2_w8_metal(ContextHandle   ctx,
                                   uint64_t*       tree,
                                   const uint64_t* input,
                                   uint64_t        num_rows,
                                   const uint64_t* C,    // length 86
                                   const uint64_t* D);   // length 8

// Arity-3 Merkle tree built with Poseidon2 W=12. `num_rows` must be a
// power of 3. `input` holds num_rows * RATE=8 u64s (one RATE-sized leaf
// input per row). `tree` is filled with (arity*num_rows - 1)/(arity-1) =
// (3*num_rows - 1)/2 nodes × CAPACITY=4 u64s. Bit-exact with
// Poseidon2Goldilocks<12>::merkletree(tree, input, RATE, num_rows, 3,
// Scalar, 1, 1).
void merkletree_poseidon2_w12_metal(ContextHandle   ctx,
                                    uint64_t*       tree,
                                    const uint64_t* input,
                                    uint64_t        num_rows,
                                    const uint64_t* C,   // length 118
                                    const uint64_t* D);  // length 12

// Arity-4 Merkle tree built with Poseidon2 W=16. `num_rows` must be a
// power of 4. Input is num_rows * RATE=12. Tree has (4*num_rows - 1)/3
// nodes × CAPACITY=4 u64s. Bit-exact with Poseidon2Goldilocks<16>::merkletree.
void merkletree_poseidon2_w16_metal(ContextHandle   ctx,
                                    uint64_t*       tree,
                                    const uint64_t* input,
                                    uint64_t        num_rows,
                                    const uint64_t* C,   // length 150
                                    const uint64_t* D);  // length 16

// Arity-{2,3,4} Merkle trees built with Poseidon2 W={8,12,16} over
// leaves of arbitrary `num_cols` u64s each. The leaf layer runs a
// sponge absorb (bit-exact with Poseidon2Goldilocks<W>::linear_hash_seq);
// parent layers zero-pad extraZeros per level to match
// Poseidon2Goldilocks<W>::merkletree_seq. `num_rows` can be any value
// >= 1 — padding handles non-pow-of-arity cases. `input` size is
// num_rows * num_cols u64s. `tree` size follows MerkleTreeGL::
// getNumNodes(num_rows) * CAPACITY = 4 u64s.
void merkletree_poseidon2_w8_cols_metal(ContextHandle   ctx,
                                        uint64_t*       tree,
                                        const uint64_t* input,
                                        uint64_t        num_cols,
                                        uint64_t        num_rows,
                                        const uint64_t* C,   // length 86
                                        const uint64_t* D);  // length 8

void merkletree_poseidon2_w12_cols_metal(ContextHandle   ctx,
                                         uint64_t*       tree,
                                         const uint64_t* input,
                                         uint64_t        num_cols,
                                         uint64_t        num_rows,
                                         const uint64_t* C,   // length 118
                                         const uint64_t* D);  // length 12

void merkletree_poseidon2_w16_cols_metal(ContextHandle   ctx,
                                         uint64_t*       tree,
                                         const uint64_t* input,
                                         uint64_t        num_cols,
                                         uint64_t        num_rows,
                                         const uint64_t* C,   // length 150
                                         const uint64_t* D);  // length 16

} // namespace pil2::metal

#endif // PIL2_HAS_METAL

#endif // PIL2_METAL_CONTEXT_HPP
