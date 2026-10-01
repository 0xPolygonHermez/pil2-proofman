#ifndef PILFFLONK_LDE_KERNELS_HPP
#define PILFFLONK_LDE_KERNELS_HPP

#include <cstdint>

// The kernels of the LDE on the device (pilfflonk_lde.cu), the elementwise work of Lde's coset
// transforms around sppark's NTTs (pilfflonk_lde_gpu.hpp), with C linkage and the conventions of
// pilfflonk_kernels.hpp: device pointers unless said otherwise, 32-byte BN254 scalars in Montgomery
// form, the legacy default stream, nothing launched for an empty range, and a CUDA failure aborts
// the process.
//
// The powers of a base b come in two tables, as gpu_plonk_precompute_omega_tables_async computes
// them with base b and block size 256 (rapidsnark/plonk_prover.cu): blocks[k] = b^(256·k) and
// powers[t] = b^t for t < 256, so that b^i = blocks[i >> 8]·powers[i & 255].
extern "C" {

// The fold of Lde::extendCosetPart before its NTT, of nCols polynomials into parts of s points:
// column t, the lengths[t] coefficients at sources[t], scaled by the powers of the part's shift c
// and folded modulo s into the s elements at dst + t·s,
//   dst[t·s + r] = Σ_k sources[t][r + k·s]·c^(r + k·s) over r + k·s < lengths[t], for r < s,
// so dst[t·s + r] = 0 for lengths[t] <= r < s. blocks and powers are the tables of c for every
// index below min(s, lengths[t]); cS, in host memory, is c^s. sources and lengths are host arrays,
// sources of device pointers, and no source overlaps dst.
void pilfflonk_gpu_fold_by_powers(void *dst, uint64_t s, const void *const *sources, const uint64_t *lengths,
                                  uint64_t nCols, const void *blocks, const void *powers, const void *cS);

// data[i] = data[i]·b^i for i < n, in place, with the tables of b for every index below n: the
// scaling of Lde::interpolateCoset by g^-i after its inverse NTT.
void pilfflonk_gpu_mul_by_powers(void *data, uint64_t n, const void *blocks, const void *powers);

} // extern "C"

#endif
