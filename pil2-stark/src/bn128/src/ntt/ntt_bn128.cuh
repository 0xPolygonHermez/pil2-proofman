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
};

#endif // __NTT_BN128_CUH__
