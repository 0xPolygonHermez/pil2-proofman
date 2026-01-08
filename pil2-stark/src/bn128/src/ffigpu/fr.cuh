#ifndef __BN128_FFIGPU_FR_CUH__
#define __BN128_FFIGPU_FR_CUH__

#define FEATURE_BN254
# if defined(__CUDA_ARCH__) || defined(__HIPCC__)
#  include <alt_bn128.hpp>
# else
#  include "alt_bn128_nvcc_host_shim.cuh"
# endif

class BN128GPUScalarField {
public:
    struct Element {
        alt_bn128::fr_t v;
    };
    static __device__ __forceinline__ Element zero();
    static __device__ __forceinline__ void copy(Element& r, const Element& a);
    static __device__ __forceinline__ void add(Element& r, const Element& a, const Element& b);
    static __device__ __forceinline__ void sub(Element& r, const Element& a, const Element& b);
    static __device__ __forceinline__ void mul(Element& r, const Element& a, const Element& b);
    static __device__ __forceinline__ void square(Element& r, const Element& a);

    static __device__ __forceinline__ Element add(const Element& a, const Element& b);
    static __device__ __forceinline__ Element sub(const Element& a, const Element& b);
    static __device__ __forceinline__ Element mul(const Element& a, const Element& b);

};
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__device__ __forceinline__ BN128GPUScalarField::Element BN128GPUScalarField::zero() {
    Element r;
    r.v[0] = 0;
    r.v[1] = 0;
    r.v[2] = 0;
    r.v[3] = 0;
    r.v[4] = 0;
    r.v[5] = 0;
    r.v[6] = 0;
    r.v[7] = 0;
    return r;
}
__device__ __forceinline__ void BN128GPUScalarField::copy(Element& r, const Element& a) {
    r.v = a.v;
}
__device__ __forceinline__ void BN128GPUScalarField::add(Element& r, const Element& a, const Element& b) {
    r.v = a.v;
    r.v += b.v;
}
__device__ __forceinline__ void BN128GPUScalarField::sub(Element& r, const Element& a, const Element& b) {
    r.v = a.v;
    r.v -= b.v;
}
__device__ __forceinline__ void BN128GPUScalarField::mul(Element& r, const Element& a, const Element& b) {
    r.v = a.v;
    r.v *= b.v;
}
__device__ __forceinline__ BN128GPUScalarField::Element BN128GPUScalarField::add(const Element& a, const Element& b) {
    
    return {a.v + b.v};
}
__device__ __forceinline__ BN128GPUScalarField::Element BN128GPUScalarField::sub(const Element& a, const Element& b) {
    
    return {a.v - b.v};
}
__device__ __forceinline__ BN128GPUScalarField::Element BN128GPUScalarField::mul(const Element& a, const Element& b) {    
    return {a.v * b.v};
}
__device__ __forceinline__ void BN128GPUScalarField::square(Element& r, const Element& a) {
    r.v = sqr(a.v);
}
#endif
#endif // __BN128_FFIGPU_FR_CUH__