// MSM GPU implementation for BN128/BN254 curve
// Uses supranational/sppark pippenger algorithm

#include <cuda.h>

// Enable BN254 curve
#define FEATURE_BN254

#include "msm_bn128.cuh"

// Include sppark field and curve types
#include <alt_bn128.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

// Define the GPU types for MSM
typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>

#ifndef __CUDA_ARCH__

void MSM_BN128_GPU::msm(PointJacobianGPU& out,
                        const PointAffineGPU* points,
                        const BN128GPUScalarField::Element* scalars,
                        size_t npoints,
                        bool mont) {

    // The CPU types (BN128::G1Point, etc.) should have the same memory layout
    // as the GPU types (jacobian_t<fp_t>, affine_inf_t<fp_t>, fr_t)
    // This is because both use 256-bit Montgomery representation
    
    point_t* gpu_out = reinterpret_cast<point_t*>(&out);
    const affine_t* gpu_points = reinterpret_cast<const affine_t*>(points);
    const scalar_t* gpu_scalars = reinterpret_cast<const scalar_t*>(scalars);
    
    // Call sppark's mult_pippenger
    RustError err = mult_pippenger<bucket_t>(gpu_out, gpu_points, npoints, 
                                              gpu_scalars, mont);
    
    if (err.code != 0) { 
        // Handle error - for now just set output to infinity //TODO
        gpu_out->inf();
    }
}

#endif // !__CUDA_ARCH__
