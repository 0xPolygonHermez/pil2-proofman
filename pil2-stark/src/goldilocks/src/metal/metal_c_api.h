#ifndef PIL2_METAL_C_API_H
#define PIL2_METAL_C_API_H

// Phase C9 integration surface. Exposes the Metal kernel entries through a
// C linkage so they can be called from:
//   - Rust (via bindgen or manual FFI declarations)
//   - Pure C code
//   - A future starks_metal.mm (which could also call the C++ pil2::metal::
//     namespace directly, but this header is the canonical integration path)
//
// All entries are no-ops on non-macOS-arm64 platforms — they return errors
// rather than linking to the Metal framework. This lets the same binary be
// built for Linux (libstarks.a) and macOS (libstarksmetal.a) without
// conditional compilation at every call site.
//
// Thread safety: the underlying context is process-wide singleton, so calls
// from multiple threads serialise at the Metal command queue. This is
// sufficient for the current prover's dispatch pattern; lock-free
// concurrent calls are a future concern.

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Return 1 if the current build has the Metal backend compiled in and a
// Metal-capable device is available, 0 otherwise. Never crashes; safe to
// call from probe / feature-detection code.
int pil2_metal_available(void);

// Fill `out` with the Metal device name (e.g. "Apple M4 Pro"). `out_len`
// includes the null terminator. Returns the number of bytes written
// (excluding the terminator). If the backend is unavailable, writes an
// empty string and returns 0.
size_t pil2_metal_device_name(char* out, size_t out_len);

// Forward NTT. Wraps pil2::metal::ntt_forward_metal with a null context
// handle internally (the C API always uses the process singleton).
//
// `data`      — in/out, length size * ncols u64.
// `size`      — power of 2.
// `ncols`     — row-major stride.
// `roots`     — (roots_len)-th roots of unity table, length roots_len
//               (power of 2, >= size).
// Returns 0 on success, non-zero if the backend is unavailable.
int pil2_metal_ntt_forward(uint64_t* data,
                           uint64_t size,
                           uint64_t ncols,
                           const uint64_t* roots,
                           uint64_t roots_len);

// Inverse NTT. `inv_n` = inv(size) in the Goldilocks field.
int pil2_metal_ntt_inverse(uint64_t* data,
                           uint64_t size,
                           uint64_t ncols,
                           const uint64_t* roots,
                           uint64_t roots_len,
                           uint64_t inv_n);

// Low-Degree Extension. `r_` is the length-N table r_[i] = shift^i / N,
// matching NTT_Goldilocks::computeR.
int pil2_metal_lde(uint64_t* output,
                   const uint64_t* input,
                   uint64_t N_Extended,
                   uint64_t N,
                   uint64_t ncols,
                   const uint64_t* roots,
                   uint64_t roots_len,
                   const uint64_t* r_);

// Poseidon2 permutation for W=8 / W=12 / W=16 over a batch of states. C
// and D are the per-width round constants and internal-round diagonals
// (see metal_context.hpp for lengths — 86/118/150 and 8/12/16).
int pil2_metal_poseidon2_permute_w8 (uint64_t* out_states, const uint64_t* in_states,
                                     uint64_t count, const uint64_t* C, const uint64_t* D);
int pil2_metal_poseidon2_permute_w12(uint64_t* out_states, const uint64_t* in_states,
                                     uint64_t count, const uint64_t* C, const uint64_t* D);
int pil2_metal_poseidon2_permute_w16(uint64_t* out_states, const uint64_t* in_states,
                                     uint64_t count, const uint64_t* C, const uint64_t* D);

// Poseidon2 Merkle tree builder. `num_rows` must be a power of the
// implicit arity (2 for W=8, 3 for W=12, 4 for W=16). `input` size is
// num_rows * RATE (4/8/12). `tree` size is (arity*num_rows-1)/(arity-1)
// × CAPACITY=4 u64s; the last CAPACITY u64s are the root.
int pil2_metal_merkletree_w8 (uint64_t* tree, const uint64_t* input, uint64_t num_rows,
                              const uint64_t* C, const uint64_t* D);
int pil2_metal_merkletree_w12(uint64_t* tree, const uint64_t* input, uint64_t num_rows,
                              const uint64_t* C, const uint64_t* D);
int pil2_metal_merkletree_w16(uint64_t* tree, const uint64_t* input, uint64_t num_rows,
                              const uint64_t* C, const uint64_t* D);

// Arity-{2,3,4} Merkle trees over leaves of arbitrary `num_cols` u64s
// each, using a sponge absorb at the leaf layer. Bit-exact with
// Poseidon2Goldilocks<W>::merkletree for any num_rows >= 1 (per-level
// zero-padding matches the CPU reference). `input` size is num_rows *
// num_cols u64s. `tree` sized per MerkleTreeGL::getNumNodes * CAPACITY=4
// u64s.
int pil2_metal_merkletree_w8_cols (uint64_t* tree, const uint64_t* input,
                                   uint64_t num_cols, uint64_t num_rows,
                                   const uint64_t* C, const uint64_t* D);
int pil2_metal_merkletree_w12_cols(uint64_t* tree, const uint64_t* input,
                                   uint64_t num_cols, uint64_t num_rows,
                                   const uint64_t* C, const uint64_t* D);
int pil2_metal_merkletree_w16_cols(uint64_t* tree, const uint64_t* input,
                                   uint64_t num_cols, uint64_t num_rows,
                                   const uint64_t* C, const uint64_t* D);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // PIL2_METAL_C_API_H
