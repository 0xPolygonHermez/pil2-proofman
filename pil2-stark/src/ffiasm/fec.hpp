#ifndef __FEC_H
#define __FEC_H

#include <stdint.h>
#include <string>
#include <gmp.h>
#include <iostream>
#include <cassert>

#define Fec_N64 4
#define Fec_SHORT 0x00000000
#define Fec_LONG 0x80000000
#define Fec_LONGMONTGOMERY 0xC0000000
typedef uint64_t FecRawElement[Fec_N64];
typedef struct __attribute__((__packed__)) {
    int32_t shortVal;
    uint32_t type;
    FecRawElement longVal;
} FecElement;
typedef FecElement *PFecElement;

#ifdef __USE_ASSEMBLY__
extern FecElement Fec_q;
extern FecElement Fec_R3;
extern FecRawElement Fec_rawq;
extern FecRawElement Fec_rawR3;

extern "C" void Fec_copy(PFecElement r, PFecElement a);
extern "C" void Fec_copyn(PFecElement r, PFecElement a, int n);
extern "C" void Fec_add(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_sub(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_neg(PFecElement r, PFecElement a);
extern "C" void Fec_mul(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_square(PFecElement r, PFecElement a);
extern "C" void Fec_band(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_bor(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_bxor(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_bnot(PFecElement r, PFecElement a);
extern "C" void Fec_shl(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_shr(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_eq(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_neq(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_lt(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_gt(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_leq(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_geq(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_land(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_lor(PFecElement r, PFecElement a, PFecElement b);
extern "C" void Fec_lnot(PFecElement r, PFecElement a);
extern "C" void Fec_toNormal(PFecElement r, PFecElement a);
extern "C" void Fec_toLongNormal(PFecElement r, PFecElement a);
extern "C" void Fec_toMontgomery(PFecElement r, PFecElement a);

extern "C" int Fec_isTrue(PFecElement pE);
extern "C" int Fec_toInt(PFecElement pE);

extern "C" void Fec_rawCopy(FecRawElement pRawResult, const FecRawElement pRawA);
extern "C" void Fec_rawSwap(FecRawElement pRawResult, FecRawElement pRawA);
extern "C" void Fec_rawAdd(FecRawElement pRawResult, const FecRawElement pRawA, const FecRawElement pRawB);
extern "C" void Fec_rawSub(FecRawElement pRawResult, const FecRawElement pRawA, const FecRawElement pRawB);
extern "C" void Fec_rawNeg(FecRawElement pRawResult, const FecRawElement pRawA);
extern "C" void Fec_rawMMul(FecRawElement pRawResult, const FecRawElement pRawA, const FecRawElement pRawB);
extern "C" void Fec_rawMSquare(FecRawElement pRawResult, const FecRawElement pRawA);
extern "C" void Fec_rawMMul1(FecRawElement pRawResult, const FecRawElement pRawA, uint64_t pRawB);
extern "C" void Fec_rawToMontgomery(FecRawElement pRawResult, const FecRawElement &pRawA);
extern "C" void Fec_rawFromMontgomery(FecRawElement pRawResult, const FecRawElement &pRawA);
extern "C" int Fec_rawIsEq(const FecRawElement pRawA, const FecRawElement pRawB);
extern "C" int Fec_rawIsZero(const FecRawElement pRawB);

extern "C" void Fec_fail();

#else

extern FecElement Fec_q;
extern FecElement Fec_R3;
extern FecRawElement Fec_rawq;
extern FecRawElement Fec_rawR3;

inline void Fec_copy(PFecElement r, PFecElement a){
    std::cerr << "Fec_copy() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_copyn(PFecElement r, PFecElement a, int n){
    std::cerr << "Fec_copyn() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_add(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_add() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_sub(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_sub() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_neg(PFecElement r, PFecElement a) {
    std::cerr << "Fec_neg() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_mul(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_mul() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_square(PFecElement r, PFecElement a) {
    std::cerr << "Fec_square() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_band(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_band() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_bor(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_bor() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_bxor(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_bxor() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_bnot(PFecElement r, PFecElement a) {
    std::cerr << "Fec_bnot() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_shl(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_shl() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_shr(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_shr() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_eq(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_eq() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_neq(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_neq() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_lt(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_lt() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_gt(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_gt() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_leq(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_leq() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_geq(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_geq() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_land(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_land() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_lor(PFecElement r, PFecElement a, PFecElement b) {
    std::cerr << "Fec_lor() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_lnot(PFecElement r, PFecElement a) {
    std::cerr << "Fec_lnot() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_toNormal(PFecElement r, PFecElement a) {
    std::cerr << "Fec_toNormal() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_toLongNormal(PFecElement r, PFecElement a) {
    std::cerr << "Fec_toLongNormal() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_toMontgomery(PFecElement r, PFecElement a) {
    std::cerr << "Fec_toMontgomery() not implemented in C++ code." << std::endl;
   assert(false);
}

inline int Fec_isTrue(PFecElement pE) {
   std::cerr << "Fec_isTrue() not implemented in C++ code." << std::endl;
  assert(false);
   return 0; // Placeholder return value
}
inline int Fec_toInt(PFecElement pE) {
    std::cerr << "Fec_toInt() not implemented in C++ code." << std::endl;
   assert(false);
    return 0; // Placeholder return value
}

inline void Fec_rawCopy(FecRawElement pRawResult, const FecRawElement pRawA) {
    std::cerr << "Fec_rawCopy() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_rawSwap(FecRawElement pRawResult, FecRawElement pRawA) {
    std::cerr << "Fec_rawSwap() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_rawAdd(FecRawElement pRawResult, const FecRawElement pRawA, const FecRawElement pRawB) {
    std::cerr << "Fec_rawAdd() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_rawSub(FecRawElement pRawResult, const FecRawElement pRawA, const FecRawElement pRawB) {
    std::cerr << "Fec_rawSub() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_rawNeg(FecRawElement pRawResult, const FecRawElement pRawA) {
    std::cerr << "Fec_rawNeg() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_rawMMul(FecRawElement pRawResult, const FecRawElement pRawA, const FecRawElement pRawB) {
    std::cerr << "Fec_rawMMul() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_rawMSquare(FecRawElement pRawResult, const FecRawElement pRawA) {
    std::cerr << "Fec_rawMSquare() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_rawMMul1(FecRawElement pRawResult, const FecRawElement pRawA, uint64_t pRawB) {
    std::cerr << "Fec_rawMMul1() not implemented in C++ code." << std::endl;
   assert(false);
}
inline void Fec_rawToMontgomery(FecRawElement pRawResult, const FecRawElement &pRawA) {
    std::cerr << "Fec_rawToMontgomery() not implemented in C++ code." << std::endl;
    assert(false);
}
inline void Fec_rawFromMontgomery(FecRawElement pRawResult, const FecRawElement &pRawA) {
    std::cerr << "Fec_rawFromMontgomery() not implemented in C++ code." << std::endl;
   assert(false);
}
inline int Fec_rawIsEq(const FecRawElement pRawA, const FecRawElement pRawB) {
    std::cerr << "Fec_rawIsEq() not implemented in C++ code." << std::endl;
   assert(false);
    return 0; // Placeholder return value
}
inline int Fec_rawIsZero(const FecRawElement pRawB) {
    std::cerr << "Fec_rawIsZero() not implemented in C++ code." << std::endl;
   assert(false);
    return 0; // Placeholder return value
}
inline void Fec_fail() {
   assert(false);
}
#endif





// Pending functions to convert

void Fec_str2element(PFecElement pE, char const*s);
char *Fec_element2str(PFecElement pE);
void Fec_idiv(PFecElement r, PFecElement a, PFecElement b);
void Fec_mod(PFecElement r, PFecElement a, PFecElement b);
void Fec_inv(PFecElement r, PFecElement a);
void Fec_div(PFecElement r, PFecElement a, PFecElement b);
void Fec_pow(PFecElement r, PFecElement a, PFecElement b);

class RawFecP {

public:
    const static int N64 = Fec_N64;
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

    void inline copy(Element &r, const Element &a) { Fec_rawCopy(r.v, a.v); };
    void inline swap(Element &a, Element &b) { Fec_rawSwap(a.v, b.v); };
    void inline add(Element &r, const Element &a, const Element &b) { Fec_rawAdd(r.v, a.v, b.v); };
    void inline sub(Element &r, const Element &a, const Element &b) { Fec_rawSub(r.v, a.v, b.v); };
    void inline mul(Element &r, const Element &a, const Element &b) { Fec_rawMMul(r.v, a.v, b.v); };

    Element inline add(const Element &a, const Element &b) { Element r; Fec_rawAdd(r.v, a.v, b.v); return r;};
    Element inline sub(const Element &a, const Element &b) { Element r; Fec_rawSub(r.v, a.v, b.v); return r;};
    Element inline mul(const Element &a, const Element &b) { Element r; Fec_rawMMul(r.v, a.v, b.v); return r;};

    Element inline neg(const Element &a) { Element r; Fec_rawNeg(r.v, a.v); return r; };
    Element inline square(const Element &a) { Element r; Fec_rawMSquare(r.v, a.v); return r; };

    Element inline add(int a, const Element &b) { return add(set(a), b);};
    Element inline sub(int a, const Element &b) { return sub(set(a), b);};
    Element inline mul(int a, const Element &b) { return mul(set(a), b);};

    Element inline add(const Element &a, int b) { return add(a, set(b));};
    Element inline sub(const Element &a, int b) { return sub(a, set(b));};
    Element inline mul(const Element &a, int b) { return mul(a, set(b));};
    
    void inline mul1(Element &r, const Element &a, uint64_t b) { Fec_rawMMul1(r.v, a.v, b); };
    void inline neg(Element &r, const Element &a) { Fec_rawNeg(r.v, a.v); };
    void inline square(Element &r, const Element &a) { Fec_rawMSquare(r.v, a.v); };
    void inv(Element &r, const Element &a);
    void div(Element &r, const Element &a, const Element &b);
    void exp(Element &r, const Element &base, uint8_t* scalar, unsigned int scalarSize);

    void inline toMontgomery(Element &r, const Element &a) { Fec_rawToMontgomery(r.v, a.v); };
    void inline fromMontgomery(Element &r, const Element &a) { Fec_rawFromMontgomery(r.v, a.v); };
    int inline eq(const Element &a, const Element &b) { return Fec_rawIsEq(a.v, b.v); };
    int inline isZero(const Element &a) { return Fec_rawIsZero(a.v); };

    void toMpz(mpz_t r, const Element &a);
    void fromMpz(Element &a, const mpz_t r);

    int toRprBE(const Element &element, uint8_t *data, int bytes);
    int fromRprBE(Element &element, const uint8_t *data, int bytes);
    
    int bytes ( void ) { return Fec_N64 * 8; };
    
    void fromUI(Element &r, unsigned long int v);

    static RawFecP field;

};


#endif // __FEC_H



