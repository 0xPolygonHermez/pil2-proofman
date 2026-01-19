#ifndef __FEC_P_H
#define __FEC_P_H

#include <stdint.h>
#include <string>
#include <gmp.h>
#include <iostream>
#include <cassert>

#define FecP_N64 4
#define FecP_SHORT 0x00000000
#define FecP_LONG 0x80000000
#define FecP_LONGMONTGOMERY 0xC0000000
typedef uint64_t FecRawElement[FecP_N64];
typedef struct __attribute__((__packed__)) {
    int32_t shortVal;
    uint32_t type;
    FecRawElement longVal;
} FecElement;
typedef FecElement *PFecElement;

#ifdef __USE_ASSEMBLY__
extern FecElement FecP_q;
extern FecElement FecP_R3;
extern FecRawElement FecP_rawq;
extern FecRawElement FecP_rawR3;

extern "C" void FecP_copy(PFecElement r, PFecElement a);
extern "C" void FecP_copyn(PFecElement r, PFecElement a, int n);
extern "C" void FecP_add(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_sub(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_neg(PFecElement r, PFecElement a);
extern "C" void FecP_mul(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_square(PFecElement r, PFecElement a);
extern "C" void FecP_band(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_bor(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_bxor(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_bnot(PFecElement r, PFecElement a);
extern "C" void FecP_shl(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_shr(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_eq(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_neq(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_lt(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_gt(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_leq(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_geq(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_land(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_lor(PFecElement r, PFecElement a, PFecElement b);
extern "C" void FecP_lnot(PFecElement r, PFecElement a);
extern "C" void FecP_toNormal(PFecElement r, PFecElement a);
extern "C" void FecP_toLongNormal(PFecElement r, PFecElement a);
extern "C" void FecP_toMontgomery(PFecElement r, PFecElement a);

extern "C" int FecP_isTrue(PFecElement pE);
extern "C" int FecP_toInt(PFecElement pE);

extern "C" void FecP_rawCopy(FecRawElement pRawResult, const FecRawElement pRawA);
extern "C" void FecP_rawSwap(FecRawElement pRawResult, FecRawElement pRawA);
extern "C" void FecP_rawAdd(FecRawElement pRawResult, const FecRawElement pRawA, const FecRawElement pRawB);
extern "C" void FecP_rawSub(FecRawElement pRawResult, const FecRawElement pRawA, const FecRawElement pRawB);
extern "C" void FecP_rawNeg(FecRawElement pRawResult, const FecRawElement pRawA);
extern "C" void FecP_rawMMul(FecRawElement pRawResult, const FecRawElement pRawA, const FecRawElement pRawB);
extern "C" void FecP_rawMSquare(FecRawElement pRawResult, const FecRawElement pRawA);
extern "C" void FecP_rawMMul1(FecRawElement pRawResult, const FecRawElement pRawA, uint64_t pRawB);
extern "C" void FecP_rawToMontgomery(FecRawElement pRawResult, const FecRawElement &pRawA);
extern "C" void FecP_rawFromMontgomery(FecRawElement pRawResult, const FecRawElement &pRawA);
extern "C" int FecP_rawIsEq(const FecRawElement pRawA, const FecRawElement pRawB);
extern "C" int FecP_rawIsZero(const FecRawElement pRawB);

extern "C" void FecP_fail();

#else

extern FecElement FecP_q;
extern FecElement FecP_R3;
extern FecRawElement FecP_rawq;
extern FecRawElement FecP_rawR3;

inline void FecP_copy(PFecElement r, PFecElement a){
    std::cerr << "FecP_copy() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_copyn(PFecElement r, PFecElement a, int n){
    std::cerr << "FecP_copyn() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_add(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_add() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_sub(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_sub() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_neg(PFecElement r, PFecElement a) {
    std::cerr << "FecP_neg() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_mul(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_mul() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_square(PFecElement r, PFecElement a) {
    std::cerr << "FecP_square() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_band(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_band() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_bor(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_bor() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_bxor(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_bxor() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_bnot(PFecElement r, PFecElement a) {
    std::cerr << "FecP_bnot() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_shl(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_shl() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_shr(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_shr() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_eq(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_eq() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_neq(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_neq() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_lt(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_lt() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_gt(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_gt() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_leq(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_leq() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_geq(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_geq() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_land(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_land() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_lor(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "FecP_lor() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_lnot(PFecElement r, PFecElement a) {
    std::cerr << "FecP_lnot() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_toNormal(PFecElement r, PFecElement a) {
    std::cerr << "FecP_toNormal() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_toLongNormal(PFecElement r, PFecElement a) {
    std::cerr << "FecP_toLongNormal() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_toMontgomery(PFecElement r, PFecElement a) {
    std::cerr << "FecP_toMontgomery() not implemented in C++ code." << std::endl;
   assert(false);
}

inline int FecP_isTrue(PFecElement pE) {
   std::cerr << "FecP_isTrue() not implemented in C++ code." << std::endl;
  assert(false);
   return 0; // Placeholder return value
}
inline int FecP_toInt(PFecElement pE) {
    std::cerr << "FecP_toInt() not implemented in C++ code." << std::endl;
   assert(false);
    return 0; // Placeholder return value
}

inline void FecP_rawCopy(FecRawElement pRawResult, const FecRawElement pRawA) {
    std::cerr << "FecP_rawCopy() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_rawSwap(FecRawElement pRawResult, FecRawElement pRawA) {
    std::cerr << "FecP_rawSwap() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_rawAdd(FecRawElement pRawResult, const FecRawElement pRawA, const FecRawElement pRawB) {
    std::cerr << "FecP_rawAdd() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_rawSub(FecRawElement pRawResult, const FecRawElement pRawA, const FecRawElement pRawB) {
    std::cerr << "FecP_rawSub() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_rawNeg(FecRawElement pRawResult, const FecRawElement pRawA) {
    std::cerr << "FecP_rawNeg() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_rawMMul(FecRawElement pRawResult, const FecRawElement pRawA, const FecRawElement pRawB) {
    std::cerr << "FecP_rawMMul() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_rawMSquare(FecRawElement pRawResult, const FecRawElement pRawA) {
    std::cerr << "FecP_rawMSquare() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_rawMMul1(FecRawElement pRawResult, const FecRawElement pRawA, uint64_t pRawB) {
    std::cerr << "FecP_rawMMul1() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void FecP_rawToMontgomery(FecRawElement pRawResult, const FecRawElement &pRawA) {
    std::cerr << "FecP_rawToMontgomery() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void FecP_rawFromMontgomery(FecRawElement pRawResult, const FecRawElement &pRawA) {
    std::cerr << "FecP_rawFromMontgomery() not implemented in C++ code." << std::endl;
   assert(false);
}
inline int FecP_rawIsEq(const FecRawElement pRawA, const FecRawElement pRawB) {
    std::cerr << "FecP_rawIsEq() not implemented in C++ code." << std::endl;
   assert(false);
    return 0; // Placeholder return value
}
inline int FecP_rawIsZero(const FecRawElement pRawB) {
    std::cerr << "FecP_rawIsZero() not implemented in C++ code." << std::endl;
   assert(false);
    return 0; // Placeholder return value
}
inline void FecP_fail() {
   assert(false);
}
#endif





// Pending functions to convert

void FecP_str2element(PFecElement pE, char const*s);
char *FecP_element2str(PFecElement pE);
void FecP_idiv(PFecElement r, PFecElement a, PFecElement b);
void FecP_mod(PFecElement r, PFecElement a, PFecElement b);
void FecP_inv(PFecElement r, PFecElement a);
void FecP_div(PFecElement r, PFecElement a, PFecElement b);
void FecP_pow(PFecElement r, PFecElement a, PFecElement b);

class RawFecP {

public:
    const static int N64 = FecP_N64;
    const static int MaxBits = 256;


    struct Element {
        FecRawElement v;
    };

private:
    Element fZero;
    Element fOne;
    Element fNegOne;

public:

    RawFecP();
    ~RawFecP();

    const Element &zero() { return fZero; };
    const Element &one() { return fOne; };
    const Element &negOne() { return fNegOne; };
    Element set(int value);
    void set(Element &r, int value);

    void fromString(Element &r, const std::string &n, uint32_t radix = 10);
    std::string toString(const Element &a, uint32_t radix = 10);

    void inline copy(Element &r, const Element &a) { FecP_rawCopy(r.v, a.v); };
    void inline swap(Element &a, Element &b) { FecP_rawSwap(a.v, b.v); };
    void inline add(Element &r, const Element &a, const Element &b) { FecP_rawAdd(r.v, a.v, b.v); };
    void inline sub(Element &r, const Element &a, const Element &b) { FecP_rawSub(r.v, a.v, b.v); };
    void inline mul(Element &r, const Element &a, const Element &b) { FecP_rawMMul(r.v, a.v, b.v); };

    Element inline add(const Element &a, const Element &b) { Element r; FecP_rawAdd(r.v, a.v, b.v); return r;};
    Element inline sub(const Element &a, const Element &b) { Element r; FecP_rawSub(r.v, a.v, b.v); return r;};
    Element inline mul(const Element &a, const Element &b) { Element r; FecP_rawMMul(r.v, a.v, b.v); return r;};

    Element inline neg(const Element &a) { Element r; FecP_rawNeg(r.v, a.v); return r; };
    Element inline square(const Element &a) { Element r; FecP_rawMSquare(r.v, a.v); return r; };

    Element inline add(int a, const Element &b) { return add(set(a), b);};
    Element inline sub(int a, const Element &b) { return sub(set(a), b);};
    Element inline mul(int a, const Element &b) { return mul(set(a), b);};

    Element inline add(const Element &a, int b) { return add(a, set(b));};
    Element inline sub(const Element &a, int b) { return sub(a, set(b));};
    Element inline mul(const Element &a, int b) { return mul(a, set(b));};
    
    void inline mul1(Element &r, const Element &a, uint64_t b) { FecP_rawMMul1(r.v, a.v, b); };
    void inline neg(Element &r, const Element &a) { FecP_rawNeg(r.v, a.v); };
    void inline square(Element &r, const Element &a) { FecP_rawMSquare(r.v, a.v); };
    void inv(Element &r, const Element &a);
    void div(Element &r, const Element &a, const Element &b);
    void exp(Element &r, const Element &base, uint8_t* scalar, unsigned int scalarSize);

    void inline toMontgomery(Element &r, const Element &a) { FecP_rawToMontgomery(r.v, a.v); };
    void inline fromMontgomery(Element &r, const Element &a) { FecP_rawFromMontgomery(r.v, a.v); };
    int inline eq(const Element &a, const Element &b) { return FecP_rawIsEq(a.v, b.v); };
    int inline isZero(const Element &a) { return FecP_rawIsZero(a.v); };

    void toMpz(mpz_t r, const Element &a);
    void fromMpz(Element &a, const mpz_t r);

    int toRprBE(const Element &element, uint8_t *data, int bytes);
    int fromRprBE(Element &element, const uint8_t *data, int bytes);
    
    int bytes ( void ) { return FecP_N64 * 8; };
    
    void fromUI(Element &r, unsigned long int v);

    static RawFecP field;

};


#endif // __FEC_P_H



