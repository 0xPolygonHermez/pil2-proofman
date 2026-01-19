#ifndef __FR_P_H
#define __FR_P_H

#include <stdint.h>
#include <string>
#include <gmp.h>
#include <iostream>
#include <cassert>

#define FrP_N64 4
#define FrP_SHORT 0x00000000
#define FrP_LONG 0x80000000
#define FrP_LONGMONTGOMERY 0xC0000000
typedef uint64_t FrRawElement[FrP_N64];
typedef struct __attribute__((__packed__)) {
    int32_t shortVal;
    uint32_t type;
    FrRawElement longVal;
} FrElement;
typedef FrElement *PFrElement;

#ifdef __USE_ASSEMBLY__

extern FrElement FrP_q;
extern FrElement FrP_R3;
extern FrRawElement FrP_rawq;
extern FrRawElement FrP_rawR3;

extern "C" void FrP_copy(PFrElement r, PFrElement a);
extern "C" void FrP_copyn(PFrElement r, PFrElement a, int n);
extern "C" void FrP_add(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_sub(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_neg(PFrElement r, PFrElement a);
extern "C" void FrP_mul(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_square(PFrElement r, PFrElement a);
extern "C" void FrP_band(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_bor(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_bxor(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_bnot(PFrElement r, PFrElement a);
extern "C" void FrP_shl(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_shr(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_eq(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_neq(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_lt(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_gt(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_leq(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_geq(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_land(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_lor(PFrElement r, PFrElement a, PFrElement b);
extern "C" void FrP_lnot(PFrElement r, PFrElement a);
extern "C" void FrP_toNormal(PFrElement r, PFrElement a);
extern "C" void FrP_toLongNormal(PFrElement r, PFrElement a);
extern "C" void FrP_toMontgomery(PFrElement r, PFrElement a);

extern "C" int FrP_isTrue(PFrElement pE);
extern "C" int FrP_toInt(PFrElement pE);

extern "C" void FrP_rawCopy(FrRawElement pRawResult, const FrRawElement pRawA);
extern "C" void FrP_rawSwap(FrRawElement pRawResult, FrRawElement pRawA);
extern "C" void FrP_rawAdd(FrRawElement pRawResult, const FrRawElement pRawA, const FrRawElement pRawB);
extern "C" void FrP_rawSub(FrRawElement pRawResult, const FrRawElement pRawA, const FrRawElement pRawB);
extern "C" void FrP_rawNeg(FrRawElement pRawResult, const FrRawElement pRawA);
extern "C" void FrP_rawMMul(FrRawElement pRawResult, const FrRawElement pRawA, const FrRawElement pRawB);
extern "C" void FrP_rawMSquare(FrRawElement pRawResult, const FrRawElement pRawA);
extern "C" void FrP_rawMMul1(FrRawElement pRawResult, const FrRawElement pRawA, uint64_t pRawB);
extern "C" void FrP_rawToMontgomery(FrRawElement pRawResult, const FrRawElement &pRawA);
extern "C" void FrP_rawFromMontgomery(FrRawElement pRawResult, const FrRawElement &pRawA);
extern "C" int FrP_rawIsEq(const FrRawElement pRawA, const FrRawElement pRawB);
extern "C" int FrP_rawIsZero(const FrRawElement pRawB);

extern "C" void FrP_fail();

#else
// Mock implementations for macOS builds

static FrElement FrP_q;
static FrElement FrP_R3;
static FrRawElement FrP_rawq;
static FrRawElement FrP_rawR3;

inline void FrP_copy(PFrElement r, PFrElement a){
    std::cerr << "FrP_copy() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_copyn(PFrElement r, PFrElement a, int n){
    std::cerr << "FrP_copyn() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_add(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_add() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_sub(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_sub() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_neg(PFrElement r, PFrElement a){
    std::cerr << "FrP_neg() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_mul(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_mul() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_square(PFrElement r, PFrElement a){
    std::cerr << "FrP_square() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_band(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_band() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_bor(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_bor() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_bxor(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_bxor() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_bnot(PFrElement r, PFrElement a){
    std::cerr << "FrP_bnot() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_shl(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_shl() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_shr(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_shr() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_eq(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_eq() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_neq(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_neq() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_lt(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_lt() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_gt(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_gt() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_leq(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_leq() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_geq(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_geq() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_land(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_land() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_lor(PFrElement r, PFrElement a, PFrElement b){
    std::cerr << "FrP_lor() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_lnot(PFrElement r, PFrElement a){
    std::cerr << "FrP_lnot() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_toNormal(PFrElement r, PFrElement a){
    std::cerr << "FrP_toNormal() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_toLongNormal(PFrElement r, PFrElement a){
    std::cerr << "FrP_toLongNormal() not implemented for macOS." << std::endl;
    assert(false);
}
inline void FrP_toMontgomery(PFrElement r, PFrElement a){
    std::cerr << "FrP_toMontgomery() not implemented for macOS." << std::endl;
    assert(false);
}

inline int FrP_isTrue(PFrElement pE){
    std::cerr << "FrP_isTrue() not implemented for macOS." << std::endl;
    assert(false);
    return 0;
}
inline int FrP_toInt(PFrElement pE){
    std::cerr << "FrP_toInt() not implemented for macOS." << std::endl;
    assert(false);
    return 0; // Placeholder return value
}

inline void FrP_rawCopy(FrRawElement pRawResult, const FrRawElement pRawA) {
    std::cerr << "FrP_rawCopy() not implemented for macOS." << std::endl;
    assert(false);
}

inline void FrP_rawSwap(FrRawElement pRawResult, FrRawElement pRawA) {
    std::cerr << "FrP_rawSwap() not implemented for macOS." << std::endl;
    assert(false);
}

inline void FrP_rawAdd(FrRawElement pRawResult, const FrRawElement pRawA, const FrRawElement pRawB) {
    std::cerr << "FrP_rawAdd() not implemented for macOS." << std::endl;
    assert(false);
}

inline void FrP_rawSub(FrRawElement pRawResult, const FrRawElement pRawA, const FrRawElement pRawB) {
    std::cerr << "FrP_rawSub() not implemented for macOS." << std::endl;
    assert(false);
}

inline void FrP_rawNeg(FrRawElement pRawResult, const FrRawElement pRawA) {
    std::cerr << "FrP_rawNeg() not implemented for macOS." << std::endl;
    assert(false);
}

inline void FrP_rawMMul(FrRawElement pRawResult, const FrRawElement pRawA, const FrRawElement pRawB) {
    std::cerr << "FrP_rawMMul() not implemented for macOS." << std::endl;
    assert(false);
}

inline void FrP_rawMSquare(FrRawElement pRawResult, const FrRawElement pRawA) {
    std::cerr << "FrP_rawMSquare() not implemented for macOS." << std::endl;
    assert(false);
}

inline void FrP_rawMMul1(FrRawElement pRawResult, const FrRawElement pRawA, uint64_t pRawB) {
    std::cerr << "FrP_rawMMul1() not implemented for macOS." << std::endl;
    assert(false);
}

inline void FrP_rawToMontgomery(FrRawElement pRawResult, const FrRawElement &pRawA) {
    std::cerr << "FrP_rawToMontgomery() not implemented for macOS." << std::endl;
    assert(false);
}

inline void FrP_rawFromMontgomery(FrRawElement pRawResult, const FrRawElement &pRawA) {
    std::cerr << "FrP_rawFromMontgomery() not implemented for macOS." << std::endl;
    assert(false);
}

inline int FrP_rawIsEq(const FrRawElement pRawA, const FrRawElement pRawB) {
    std::cerr << "FrP_rawIsEq() not implemented for macOS." << std::endl;
    assert(false);
    return 0;
}

inline int FrP_rawIsZero(const FrRawElement pRawB) {
    std::cerr << "FrP_rawIsZero() not implemented for macOS." << std::endl;
    assert(false);
    return 0;
}

inline void FrP_fail() {
    assert(false);
}

#endif // __USE_ASSEMBLY__


// Pending functions to convert

void FrP_str2element(PFrElement pE, char const*s);
void FrP_str2element(PFrElement pE, char const *s, unsigned int base);
char *FrP_element2str(PFrElement pE);
void FrP_idiv(PFrElement r, PFrElement a, PFrElement b);
void FrP_mod(PFrElement r, PFrElement a, PFrElement b);
void FrP_inv(PFrElement r, PFrElement a);
void FrP_div(PFrElement r, PFrElement a, PFrElement b);
void FrP_pow(PFrElement r, PFrElement a, PFrElement b);

class RawFrP {

public:
    const static int N64 = FrP_N64;
    const static int MaxBits = 254;


    struct Element {
        FrRawElement v;
    };

private:
    Element fZero;
    Element fOne;
    Element fNegOne;

public:

    RawFrP();
    ~RawFrP();

    const Element &zero() { return fZero; };
    const Element &one() { return fOne; };
    const Element &negOne() { return fNegOne; };
    Element set(int value);
    void set(Element &r, int value);

    void fromString(Element &r, const std::string &n, uint32_t radix = 10);
    std::string toString(const Element &a, uint32_t radix = 10);

    void inline copy(Element &r, const Element &a) { FrP_rawCopy(r.v, a.v); };
    void inline swap(Element &a, Element &b) { FrP_rawSwap(a.v, b.v); };
    void inline add(Element &r, const Element &a, const Element &b) { FrP_rawAdd(r.v, a.v, b.v); };
    void inline sub(Element &r, const Element &a, const Element &b) { FrP_rawSub(r.v, a.v, b.v); };
    void inline mul(Element &r, const Element &a, const Element &b) { FrP_rawMMul(r.v, a.v, b.v); };

    Element inline add(const Element &a, const Element &b) { Element r; FrP_rawAdd(r.v, a.v, b.v); return r;};
    Element inline sub(const Element &a, const Element &b) { Element r; FrP_rawSub(r.v, a.v, b.v); return r;};
    Element inline mul(const Element &a, const Element &b) { Element r; FrP_rawMMul(r.v, a.v, b.v); return r;};

    Element inline neg(const Element &a) { Element r; FrP_rawNeg(r.v, a.v); return r; };
    Element inline square(const Element &a) { Element r; FrP_rawMSquare(r.v, a.v); return r; };

    Element inline add(int a, const Element &b) { return add(set(a), b);};
    Element inline sub(int a, const Element &b) { return sub(set(a), b);};
    Element inline mul(int a, const Element &b) { return mul(set(a), b);};

    Element inline add(const Element &a, int b) { return add(a, set(b));};
    Element inline sub(const Element &a, int b) { return sub(a, set(b));};
    Element inline mul(const Element &a, int b) { return mul(a, set(b));};
    
    void inline mul1(Element &r, const Element &a, uint64_t b) { FrP_rawMMul1(r.v, a.v, b); };
    void inline neg(Element &r, const Element &a) { FrP_rawNeg(r.v, a.v); };
    void inline square(Element &r, const Element &a) { FrP_rawMSquare(r.v, a.v); };
    void inv(Element &r, const Element &a);
    void div(Element &r, const Element &a, const Element &b);
    void exp(Element &r, const Element &base, uint8_t* scalar, unsigned int scalarSize);

    void inline toMontgomery(Element &r, const Element &a) { FrP_rawToMontgomery(r.v, a.v); };
    void inline fromMontgomery(Element &r, const Element &a) { FrP_rawFromMontgomery(r.v, a.v); };
    int inline eq(const Element &a, const Element &b) { return FrP_rawIsEq(a.v, b.v); };
    int inline isZero(const Element &a) { return FrP_rawIsZero(a.v); };

    void toMpz(mpz_t r, const Element &a);
    void fromMpz(Element &a, const mpz_t r);

    int toRprBE(const Element &element, uint8_t *data, int bytes);
    int fromRprBE(Element &element, const uint8_t *data, int bytes);
    int fromRprLE(Element &element, const uint8_t *data, int bytes);
    int toRprLE(const Element &element, uint8_t *data, int bytes);
    
    int bytes ( void ) { return FrP_N64 * 8; };
    
    void fromUI(Element &r, unsigned long int v);

    static RawFrP field;

};


#endif // __FR_P_H


