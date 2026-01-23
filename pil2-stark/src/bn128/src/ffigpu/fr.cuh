#ifndef __BN128_FFIGPU_FR_CUH__
#define __BN128_FFIGPU_FR_CUH__

#ifndef FEATURE_BN254
#define FEATURE_BN254
#endif
#include <alt_bn128.hpp>

// BN128 Scalar Field (Fr) 
class BN128GPUScalarField {
public:
    struct Element {
        alt_bn128::fr_t v;
    };
    static __device__ __forceinline__ Element zero();
    static __device__ __forceinline__ Element one();
    static __device__ __forceinline__ void copy(Element& r, const Element& a);
    static __device__ __forceinline__ void add(Element& r, const Element& a, const Element& b);
    static __device__ __forceinline__ void sub(Element& r, const Element& a, const Element& b);
    static __device__ __forceinline__ void mul(Element& r, const Element& a, const Element& b);
    static __device__ __forceinline__ void square(Element& r, const Element& a);

    static __device__ __forceinline__ Element add(const Element& a, const Element& b);
    static __device__ __forceinline__ Element sub(const Element& a, const Element& b);
    static __device__ __forceinline__ Element mul(const Element& a, const Element& b);

    static __device__ __forceinline__ void toMontgomery(Element& r);
    static __device__ __forceinline__ void fromMontgomery(Element& r);
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

__device__ __forceinline__ BN128GPUScalarField::Element BN128GPUScalarField::one() {
    Element r;
    // Return 1 in Montgomery form (R mod p)
    r.v = alt_bn128::fr_t::one();
    return r;
}

__device__ __forceinline__ void BN128GPUScalarField::copy(Element& r, const Element& a) {
    r.v = a.v;
}

__device__ __forceinline__ void BN128GPUScalarField::add(Element& r, const Element& a, const Element& b) {
    r.v = a.v + b.v;
}

__device__ __forceinline__ void BN128GPUScalarField::sub(Element& r, const Element& a, const Element& b) {
    r.v = a.v - b.v;
}

__device__ __forceinline__ void BN128GPUScalarField::mul(Element& r, const Element& a, const Element& b) {
    r.v = a.v * b.v;
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
__device__ __forceinline__ void BN128GPUScalarField::toMontgomery(Element& r) {
    r.v.to();
}
__device__ __forceinline__ void BN128GPUScalarField::fromMontgomery(Element& r) {
    r.v.from();
}
#endif

#endif // __BN128_FFIGPU_FR_CUH__
