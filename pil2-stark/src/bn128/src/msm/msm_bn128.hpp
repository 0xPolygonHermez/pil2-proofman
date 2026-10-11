// Multi-scalar multiplication over BN128/BN254, CPU and GPU behind one header.
//
// Plain C++ so it can be included from either side: the GPU entry point is
// reached through the extern "C" shim in msm_bn128.cu, and the CPU entry point
// is ffiasm's ParallelMultiexp. Only the dispatch and the point-format
// conversion live here -- neither implementation is duplicated.
//
// This was previously private to plonk_prover_gpu.c.cuh, which meant any other
// consumer had to reimplement the conversion below. It is the same situation
// the CPU NTT was in: an implementation that exists but is not reachable.

#ifndef MSM_BN128_HPP
#define MSM_BN128_HPP

#include <cstddef>
#include <cstdint>

// Implemented in msm_bn128.cu. Present only in GPU builds; guarded at the call
// site by whoever links it.
extern "C" void msm_bn128_gpu_dev_ptr(void* out, const void* d_points, const void* d_scalars, size_t npoints,
                                      bool montgomery);

namespace MsmBn128 {

/// MSM over scalars and points already resident on the device.
///
/// sppark returns a point in standard Jacobian coordinates (X, Y, Z), while
/// ffiasm's G1Point is extended Jacobian (x, y, zz, zzz) with zz = Z^2 and
/// zzz = Z^3. Getting that conversion wrong yields a point that is wrong but
/// still on the curve, so it fails as a bad commitment rather than a crash.
template <typename Engine>
typename Engine::G1Point msmDevPtr(Engine& E, const void* dPoints, const void* dScalars, size_t npoints) {
    struct JacobianPoint {
        typename Engine::F1Element X;
        typename Engine::F1Element Y;
        typename Engine::F1Element Z;
    };
    JacobianPoint gpuResult;

    msm_bn128_gpu_dev_ptr(&gpuResult, dPoints, dScalars, npoints, true);

    typename Engine::G1Point value;
    value.x = gpuResult.X;
    value.y = gpuResult.Y;
    E.f1.square(value.zz, gpuResult.Z);
    E.f1.mul(value.zzz, value.zz, gpuResult.Z);
    return value;
}

/// MSM on the host, via ffiasm's ParallelMultiexp.
///
/// `scalars` are raw bytes of `scalarSize` each, matching
/// Curve::multiMulByScalar -- not field elements, since the CPU path reads the
/// scalars as little-endian integers rather than in Montgomery form.
template <typename Engine>
typename Engine::G1Point msmHost(Engine& E, typename Engine::G1PointAffine* points, uint8_t* scalars,
                                 unsigned int scalarSize, unsigned int npoints, unsigned int nThreads = 0) {
    typename Engine::G1Point value;
    E.g1.multiMulByScalar(value, points, scalars, scalarSize, npoints, nThreads);
    return value;
}

} // namespace MsmBn128

#endif // MSM_BN128_HPP
