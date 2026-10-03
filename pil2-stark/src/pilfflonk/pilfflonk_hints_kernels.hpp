#ifndef PILFFLONK_HINTS_KERNELS_HPP
#define PILFFLONK_HINTS_KERNELS_HPP

#include <cstdint>

// The kernels of the std's prover hints on the device (pilfflonk_hints.cu), the elementwise work and
// the scan of a hint's column around the interpreter (pilfflonk_hints_gpu.hpp), with C linkage and
// the conventions of pilfflonk_kernels.hpp: device pointers unless said otherwise, 32-byte BN254
// scalars in Montgomery form, the legacy default stream, nothing launched for an empty range, and a
// CUDA failure aborts the process. The running product of a gprod_col is the PLONK GPU prover's
// gpu_plonk_prefix_scan_multiply (rapidsnark/plonk_prover.cu), as it is.

extern "C" {

// An operand of a hint's quotient on the n rows of H (HintInput, pilfflonk_proving_key.hpp): on row
// i, values[(i + shift) mod n] of a column of n values on the device (an expression's, computed
// there, has shift 0), or `number` (Montgomery form, host memory in the struct) if values is null.
struct HintOperand {
    const void *values;
    uint64_t shift;
    uint64_t number[4];
};

// The quotient of a hint on the n rows of H, n a power of two: dest[i] = numerator(i)·denominator(i)^−1,
// as Instance::computeHintColumns computes it with a batch inversion, and *firstZero the first row
// where the denominator is 0, or UINT64_MAX if there is none (dest[i] is then 0 on such a row).
// numerator.values may be dest itself, at shift 0: each row is read before it is written. The
// operands are host pointers; firstZero is device memory.
void pilfflonk_gpu_hint_quotient(void *dest, uint64_t n, const HintOperand *numerator, const HintOperand *denominator,
                                 uint64_t *firstZero);

// The running sum of a gsum_col, in place: data[i] = data[0] + … + data[i] for i < n, as
// gpu_plonk_prefix_scan_multiply computes the running product, with `work` of the same size
// (pilfflonk_gpu_prefix_scan_work_elements).
void pilfflonk_gpu_prefix_scan_add(void *data, uint64_t n, void *work);

// The elements of `work` that a scan of n elements takes, gpu_plonk_prefix_scan_multiply's and
// pilfflonk_gpu_prefix_scan_add's: the totals of the blocks of 1024 elements of each level of the
// recursion but the last, which fits in one block.
uint64_t pilfflonk_gpu_prefix_scan_work_elements(uint64_t n);

} // extern "C"

#endif
