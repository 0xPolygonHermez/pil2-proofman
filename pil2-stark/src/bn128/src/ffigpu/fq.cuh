#ifndef __BN128_FFIGPU_FQ_CUH__
#define __BN128_FFIGPU_FQ_CUH__

#ifndef FEATURE_BN254
#define FEATURE_BN254
#endif
#include <alt_bn128.hpp>

// BN128 Base Field (Fq) - used for curve point coordinates
class BN128GPUBaseField {
public:
    struct Element {
        alt_bn128::fp_t v;
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

    static __device__ __forceinline__ void toMongomery(Element& r);
    static __device__ __forceinline__ void fromMongomery(Element& r);
};

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__device__ __forceinline__ BN128GPUBaseField::Element BN128GPUBaseField::zero() {
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

__device__ __forceinline__ BN128GPUBaseField::Element BN128GPUBaseField::one() {
    Element r;
    // Return 1 in Montgomery form (R mod p)
    r.v = alt_bn128::fp_t::one();
    return r;
}

__device__ __forceinline__ void BN128GPUBaseField::copy(Element& r, const Element& a) {
    r.v = a.v;
}

__device__ __forceinline__ void BN128GPUBaseField::add(Element& r, const Element& a, const Element& b) {
    r.v = a.v + b.v;
}

__device__ __forceinline__ void BN128GPUBaseField::sub(Element& r, const Element& a, const Element& b) {
    r.v = a.v - b.v;
}

__device__ __forceinline__ void BN128GPUBaseField::mul(Element& r, const Element& a, const Element& b) {
    r.v = a.v * b.v;
}

__device__ __forceinline__ BN128GPUBaseField::Element BN128GPUBaseField::add(const Element& a, const Element& b) {
    return {a.v + b.v};
}

__device__ __forceinline__ BN128GPUBaseField::Element BN128GPUBaseField::sub(const Element& a, const Element& b) {
    return {a.v - b.v};
}

__device__ __forceinline__ BN128GPUBaseField::Element BN128GPUBaseField::mul(const Element& a, const Element& b) {
    return {a.v * b.v};
}

__device__ __forceinline__ void BN128GPUBaseField::square(Element& r, const Element& a) {
    r.v = sqr(a.v);
}

__device__ __forceinline__ void BN128GPUBaseField::toMongomery(Element& r) {
    r.v.to();
}

__device__ __forceinline__ void BN128GPUBaseField::fromMongomery(Element& r) {
    r.v.from();
}
#endif

#endif // __BN128_FFIGPU_FQ_CUH__
