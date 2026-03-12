// MSM GPU implementation for BN128/BN254 curve
// Uses supranational/sppark pippenger algorithm

#include <cuda.h>

// Enable BN254 curve
#ifndef FEATURE_BN254
#define FEATURE_BN254
#endif

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

// C-linkage wrapper function for calling from g++ code
// This allows plonk_prover_gpu (compiled with g++) to call GPU MSM
extern "C" void msm_bn128_gpu(void* out, const void* points, const void* scalars, size_t npoints, bool mont) {
    
    point_t* gpu_out = reinterpret_cast<point_t*>(out);
    const affine_t* gpu_points = reinterpret_cast<const affine_t*>(points);
    const scalar_t* gpu_scalars = reinterpret_cast<const scalar_t*>(scalars);
    
    RustError err = mult_pippenger<bucket_t>(gpu_out, gpu_points, npoints, 
                                              gpu_scalars, mont);
    
    if (err.code != 0) { 
        gpu_out->inf();
    }
}

// Device pointer version 
extern "C" void msm_bn128_gpu_dev_ptr(void* out, const void* d_points, const void* d_scalars, size_t npoints, bool mont) {
    
    point_t* host_out = reinterpret_cast<point_t*>(out);
    
    // Wrap device pointers in dev_ptr_t - sppark detects is_device_ptr and skips HtoD copy
    dev_ptr_t<affine_t> gpu_points(const_cast<affine_t*>(reinterpret_cast<const affine_t*>(d_points)), npoints);
    dev_ptr_t<scalar_t> gpu_scalars(const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(d_scalars)), npoints);
    
    try {
        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, npoints};
        RustError err = msm.invoke(*host_out, gpu_points, npoints, gpu_scalars, mont);
        
        if (err.code != 0) {
            host_out->inf();
        }
    } catch (const cuda_error& e) {
        host_out->inf();
    }
}

#endif // !__CUDA_ARCH__
