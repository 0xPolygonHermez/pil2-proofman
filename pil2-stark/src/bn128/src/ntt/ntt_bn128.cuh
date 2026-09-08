// NTT (Number Theoretic Transform) GPU implementation for BN128/BN254 scalar field
// Wrapper around supranational/sppark NTT implementation

#ifndef __NTT_BN128_CUH__
#define __NTT_BN128_CUH__

#include "bn128.cuh"
#include <cstddef>
#include <cstdint>

// NTT GPU interface
class NTT_BN128_GPU {
public:

    static void ntt(BN128GPUScalarField::Element* data, uint32_t lg_n);
    static void intt(BN128GPUScalarField::Element* data, uint32_t lg_n);

    // Low degree extension, matching NTT_AltBn128::extendPol on the CPU:
    // interpolate over the domain of size 2^lg_n, zero-pad the coefficients,
    // then evaluate over the domain of size 2^(lg_n + lg_blowup).
    //
    // `data` must hold 2^(lg_n + lg_blowup) elements. On entry its first 2^lg_n
    // hold the evaluations to extend; on return the whole buffer holds the
    // extended evaluations.
    //
    // NOTE: this is deliberately NOT sppark's NTT::LDE(). That entry point
    // multiplies the coefficients by generator powers before the second
    // transform, evaluating over a *coset* rather than over the larger
    // subgroup, so it does not agree with extendPol. Use lde_coset() when a
    // coset is what you want.
    static void lde(BN128GPUScalarField::Element* data, uint32_t lg_n, uint32_t lg_blowup);

    // Coset low degree extension: sppark's NTT::LDE() semantics, evaluating
    // over a coset of the extended domain. Has no NTT_AltBn128 counterpart.
    static void lde_coset(BN128GPUScalarField::Element* data, uint32_t lg_n, uint32_t lg_blowup);

    // Multi-column variants. The CPU stores columns interleaved row-major
    // (element (row, col) at index row * ncols + col), which sppark cannot
    // consume directly -- it transforms one contiguous column at a time -- so
    // these de-interleave, transform per column, and re-interleave.
    static void ntt_multicol(BN128GPUScalarField::Element* data, uint32_t lg_n, uint32_t ncols);
    static void intt_multicol(BN128GPUScalarField::Element* data, uint32_t lg_n, uint32_t ncols);

    // Multi-column LDE. `data` holds 2^(lg_n + lg_blowup) * ncols elements,
    // interleaved; the first 2^lg_n rows carry the input.
    static void lde_multicol(BN128GPUScalarField::Element* data, uint32_t lg_n, uint32_t lg_blowup,
                             uint32_t ncols);
};

#endif // __NTT_BN128_CUH__
