#ifndef __FQ_P_H
#define __FQ_P_H

#include <stdint.h>
#include <string>
#include <gmp.h>
#include <iostream>
#include <cassert>

#define FqP_N64 4
#define FqP_SHORT 0x00000000
#define FqP_LONG 0x80000000
#define FqP_LONGMONTGOMERY 0xC0000000
typedef uint64_t FqRawElement[FqP_N64];
typedef struct __attribute__((__packed__)) {
    int32_t shortVal;
    uint32_t type;
    FqRawElement longVal;
} FqElement;
typedef FqElement *PFqElement;

#ifdef __USE_ASSEMBLY__
extern FqElement FqP_q;
extern FqElement FqP_R3;
extern FqRawElement FqP_rawq;
extern FqRawElement FqP_rawR3;

extern "C" void FqP_copy(PFqElement r, PFqElement a);
extern "C" void FqP_copyn(PFqElement r, PFqElement a, int n);
extern "C" void FqP_add(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_sub(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_neg(PFqElement r, PFqElement a);
extern "C" void FqP_mul(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_square(PFqElement r, PFqElement a);
extern "C" void FqP_band(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_bor(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_bxor(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_bnot(PFqElement r, PFqElement a);
extern "C" void FqP_shl(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_shr(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_eq(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_neq(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_lt(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_gt(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_leq(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_geq(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_land(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_lor(PFqElement r, PFqElement a, PFqElement b);
extern "C" void FqP_lnot(PFqElement r, PFqElement a);
extern "C" void FqP_toNormal(PFqElement r, PFqElement a);
extern "C" void FqP_toLongNormal(PFqElement r, PFqElement a);
extern "C" void FqP_toMontgomery(PFqElement r, PFqElement a);

extern "C" int FqP_isTrue(PFqElement pE);
extern "C" int FqP_toInt(PFqElement pE);

extern "C" void FqP_rawCopy(FqRawElement pRawResult, const FqRawElement pRawA);
extern "C" void FqP_rawSwap(FqRawElement pRawResult, FqRawElement pRawA);
extern "C" void FqP_rawAdd(FqRawElement pRawResult, const FqRawElement pRawA, const FqRawElement pRawB);
extern "C" void FqP_rawSub(FqRawElement pRawResult, const FqRawElement pRawA, const FqRawElement pRawB);
extern "C" void FqP_rawNeg(FqRawElement pRawResult, const FqRawElement pRawA);
extern "C" void FqP_rawMMul(FqRawElement pRawResult, const FqRawElement pRawA, const FqRawElement pRawB);
extern "C" void FqP_rawMSquare(FqRawElement pRawResult, const FqRawElement pRawA);
extern "C" void FqP_rawMMul1(FqRawElement pRawResult, const FqRawElement pRawA, uint64_t pRawB);
extern "C" void FqP_rawToMontgomery(FqRawElement pRawResult, const FqRawElement &pRawA);
extern "C" void FqP_rawFromMontgomery(FqRawElement pRawResult, const FqRawElement &pRawA);
extern "C" int FqP_rawIsEq(const FqRawElement pRawA, const FqRawElement pRawB);
extern "C" int FqP_rawIsZero(const FqRawElement pRawB);

extern "C" void FqP_fail();
#else
static FqElement FqP_q;
static FqElement FqP_R3;
static FqRawElement FqP_rawq;
static FqRawElement FqP_rawR3;

inline void FqP_copy(PFqElement r, PFqElement a){
    std::cerr << "FqP_copy() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_copyn(PFqElement r, PFqElement a, int n){
    std::cerr << "FqP_copyn() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_add(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_add() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_sub(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_sub() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_neg(PFqElement r, PFqElement a){
    std::cerr << "FqP_neg() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_mul(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_mul() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_square(PFqElement r, PFqElement a){
    std::cerr << "FqP_square() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_band(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_band() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_bor(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_bor() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_bxor(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_bxor() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_bnot(PFqElement r, PFqElement a){
    std::cerr << "FqP_bnot() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_shl(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_shl() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_shr(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_shr() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_eq(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_eq() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_neq(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_neq() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_lt(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_lt() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_gt(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_gt() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_leq(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_leq() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_geq(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_geq() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_land(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_land() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_lor(PFqElement r, PFqElement a, PFqElement b){
    std::cerr << "FqP_lor() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_lnot(PFqElement r, PFqElement a){
    std::cerr << "FqP_lnot() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_toNormal(PFqElement r, PFqElement a){
    std::cerr << "FqP_toNormal() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_toLongNormal(PFqElement r, PFqElement a){
    std::cerr << "FqP_toLongNormal() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_toMontgomery(PFqElement r, PFqElement a){
    std::cerr << "FqP_toMontgomery() not implemented in C++ code." << std::endl;
    assert(false);
}
inline int FqP_isTrue(PFqElement pE){
    std::cerr << "FqP_isTrue() not implemented in C++ code." << std::endl;
    assert(false);
    return 0; // Placeholder return value
}
inline int FqP_toInt(PFqElement pE){
    std::cerr << "FqP_toInt() not implemented in C++ code." << std::endl;
    assert(false);
    return 0; // Placeholder return value
}
inline void FqP_rawCopy(FqRawElement pRawResult, const FqRawElement pRawA){
    std::cerr << "FqP_rawCopy() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_rawSwap(FqRawElement pRawResult, FqRawElement pRawA){
    std::cerr << "FqP_rawSwap() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_rawAdd(FqRawElement pRawResult, const FqRawElement pRawA, const FqRawElement pRawB){
    std::cerr << "FqP_rawAdd() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_rawSub(FqRawElement pRawResult, const FqRawElement pRawA, const FqRawElement pRawB){
    std::cerr << "FqP_rawSub() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_rawNeg(FqRawElement pRawResult, const FqRawElement pRawA){
    std::cerr << "FqP_rawNeg() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_rawMMul(FqRawElement pRawResult, const FqRawElement pRawA, const FqRawElement pRawB){
    std::cerr << "FqP_rawMMul() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_rawMSquare(FqRawElement pRawResult, const FqRawElement pRawA){
    std::cerr << "FqP_rawMSquare() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_rawMMul1(FqRawElement pRawResult, const FqRawElement pRawA, uint64_t pRawB){
    std::cerr << "FqP_rawMMul1() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_rawToMontgomery(FqRawElement pRawResult, const FqRawElement &pRawA){
    std::cerr << "FqP_rawToMontgomery() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FqP_rawFromMontgomery(FqRawElement pRawResult, const FqRawElement &pRawA){
    std::cerr << "FqP_rawFromMontgomery() not implemented in C++ code." << std::endl;
    assert(false);
}
inline int FqP_rawIsEq(const FqRawElement pRawA, const FqRawElement pRawB){
    std::cerr << "FqP_rawIsEq() not implemented in C++ code." << std::endl;
    assert(false);
    return 0; // Placeholder return value
}
inline int FqP_rawIsZero(const FqRawElement pRawB){
    std::cerr << "FqP_rawIsZero() not implemented in C++ code." << std::endl;
    assert(false);
    return 0; // Placeholder return value
}
inline void FqP_fail() {
    assert(false);
}
#endif


// Pending functions to convert

void FqP_str2element(PFqElement pE, char const*s);
char *FqP_element2str(PFqElement pE);
void FqP_idiv(PFqElement r, PFqElement a, PFqElement b);
void FqP_mod(PFqElement r, PFqElement a, PFqElement b);
void FqP_inv(PFqElement r, PFqElement a);
void FqP_div(PFqElement r, PFqElement a, PFqElement b);
void FqP_pow(PFqElement r, PFqElement a, PFqElement b);

class RawFqP {

public:
    const static int N64 = FqP_N64;
    const static int MaxBits = 254;


    struct Element {
        FqRawElement v;
    };

private:
    Element fZero;
    Element fOne;
    Element fNegOne;

public:

    RawFqP();
    ~RawFqP();

    const Element &zero() { return fZero; };
    const Element &one() { return fOne; };
    const Element &negOne() { return fNegOne; };
    Element set(int value);
    void set(Element &r, int value);

    void fromString(Element &r, const std::string &n, uint32_t radix = 10);
    std::string toString(const Element &a, uint32_t radix = 10);

    void inline copy(Element &r, const Element &a) { FqP_rawCopy(r.v, a.v); };
    void inline swap(Element &a, Element &b) { FqP_rawSwap(a.v, b.v); };
    void inline add(Element &r, const Element &a, const Element &b) { FqP_rawAdd(r.v, a.v, b.v); };
    void inline sub(Element &r, const Element &a, const Element &b) { FqP_rawSub(r.v, a.v, b.v); };
    void inline mul(Element &r, const Element &a, const Element &b) { FqP_rawMMul(r.v, a.v, b.v); };

    Element inline add(const Element &a, const Element &b) { Element r; FqP_rawAdd(r.v, a.v, b.v); return r;};
    Element inline sub(const Element &a, const Element &b) { Element r; FqP_rawSub(r.v, a.v, b.v); return r;};
    Element inline mul(const Element &a, const Element &b) { Element r; FqP_rawMMul(r.v, a.v, b.v); return r;};

    Element inline neg(const Element &a) { Element r; FqP_rawNeg(r.v, a.v); return r; };
    Element inline square(const Element &a) { Element r; FqP_rawMSquare(r.v, a.v); return r; };

    Element inline add(int a, const Element &b) { return add(set(a), b);};
    Element inline sub(int a, const Element &b) { return sub(set(a), b);};
    Element inline mul(int a, const Element &b) { return mul(set(a), b);};

    Element inline add(const Element &a, int b) { return add(a, set(b));};
    Element inline sub(const Element &a, int b) { return sub(a, set(b));};
    Element inline mul(const Element &a, int b) { return mul(a, set(b));};
    
    void inline mul1(Element &r, const Element &a, uint64_t b) { FqP_rawMMul1(r.v, a.v, b); };
    void inline neg(Element &r, const Element &a) { FqP_rawNeg(r.v, a.v); };
    void inline square(Element &r, const Element &a) { FqP_rawMSquare(r.v, a.v); };
    void inv(Element &r, const Element &a);
    void div(Element &r, const Element &a, const Element &b);
    void exp(Element &r, const Element &base, uint8_t* scalar, unsigned int scalarSize);

    void inline toMontgomery(Element &r, const Element &a) { FqP_rawToMontgomery(r.v, a.v); };
    void inline fromMontgomery(Element &r, const Element &a) { FqP_rawFromMontgomery(r.v, a.v); };
    int inline eq(const Element &a, const Element &b) { return FqP_rawIsEq(a.v, b.v); };
    int inline isZero(const Element &a) { return FqP_rawIsZero(a.v); };

    void toMpz(mpz_t r, const Element &a);
    void fromMpz(Element &a, const mpz_t r);

    int toRprBE(const Element &element, uint8_t *data, int bytes);
    int fromRprBE(Element &element, const uint8_t *data, int bytes);
    
    int toRprLE(const Element &element, uint8_t *data, int bytes);
    int fromRprLE(Element &element, const uint8_t *data, int bytes);
    
    int bytes ( void ) { return FqP_N64 * 8; };
    
    void fromUI(Element &r, unsigned long int v);

    static RawFqP field;

};


#endif // __FQ_P_H



