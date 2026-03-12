#ifndef __BN128_CURVE_POINT_CUH__
#define __BN128_CURVE_POINT_CUH__

#include "fq.cuh"

// GPU Point types for BN128 curve
// Designed to be compatible with sppark's jacobian_t and Affine_inf_t types.

// Affine point representation (x, y)
// For infinity point, both x and y are zero
struct PointAffineGPU {
    BN128GPUBaseField::Element x;
    BN128GPUBaseField::Element y;
};

// Affine point with explicit infinity flag
// Compatible with sppark's Affine_inf_t<fp_t>
struct PointAffineInfGPU {
    BN128GPUBaseField::Element x;
    BN128GPUBaseField::Element y;
    bool inf;
};

// Jacobian point representation (X, Y, Z)
// Represents affine point (X/Z^2, Y/Z^3)
// Infinity when Z = 0
// Compatible with sppark's jacobian_t<fp_t>
struct PointJacobianGPU {
    BN128GPUBaseField::Element X;
    BN128GPUBaseField::Element Y;
    BN128GPUBaseField::Element Z;
};

#endif // __BN128_CURVE_POINT_CUH__
