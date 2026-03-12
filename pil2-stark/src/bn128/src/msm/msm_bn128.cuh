// MSM (Multi-Scalar Multiplication) GPU implementation for BN128/BN254 curve
// Wrapper around supranational/sppark pippenger implementation

#ifndef __MSM_BN128_CUH__
#define __MSM_BN128_CUH__

#include "bn128.cuh"
#include <cstddef>
#include "point.cuh"

// MSM GPU interface
class MSM_BN128_GPU {
public:
   
    static void msm(PointJacobianGPU& out,
                    const PointAffineGPU* points,
                    const BN128GPUScalarField::Element* scalars,
                    size_t npoints,
                    bool mont = false);
};

#endif // __MSM_BN128_CUH__
