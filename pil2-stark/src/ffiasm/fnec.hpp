#ifndef __FNEC_P_H
#define __FNEC_P_H

#include <stdint.h>
#include <string>
#include <gmp.h>
#include <iostream>
#include <cassert>

#define FnecP_N64 4
#define FnecP_SHORT 0x00000000
#define FnecP_LONG 0x80000000
#define FnecP_LONGMONTGOMERY 0xC0000000
typedef uint64_t FnecRawElement[FnecP_N64];
typedef struct __attribute__((__packed__)) {
    int32_t shortVal;
    uint32_t type;
    FnecRawElement longVal;
} FnecElement;
typedef FnecElement *PFnecElement;

#ifdef __USE_ASSEMBLY__
    extern FnecElement FnecP_q;
    extern FnecElement FnecP_R3;
    extern FnecRawElement FnecP_rawq;
    extern FnecRawElement FnecP_rawR3;

    extern "C" void FnecP_copy(PFnecElement r, PFnecElement a);
    extern "C" void FnecP_copyn(PFnecElement r, PFnecElement a, int n);
    extern "C" void FnecP_add(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_sub(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_neg(PFnecElement r, PFnecElement a);
    extern "C" void FnecP_mul(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_square(PFnecElement r, PFnecElement a);
    extern "C" void FnecP_band(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_bor(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_bxor(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_bnot(PFnecElement r, PFnecElement a);
    extern "C" void FnecP_shl(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_shr(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_eq(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_neq(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_lt(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_gt(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_leq(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_geq(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_land(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_lor(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void FnecP_lnot(PFnecElement r, PFnecElement a);
    extern "C" void FnecP_toNormal(PFnecElement r, PFnecElement a);
    extern "C" void FnecP_toLongNormal(PFnecElement r, PFnecElement a);
    extern "C" void FnecP_toMontgomery(PFnecElement r, PFnecElement a);

    extern "C" int FnecP_isTrue(PFnecElement pE);
    extern "C" int FnecP_toInt(PFnecElement pE);

    extern "C" void FnecP_rawCopy(FnecRawElement pRawResult, const FnecRawElement pRawA);
    extern "C" void FnecP_rawSwap(FnecRawElement pRawResult, FnecRawElement pRawA);
    extern "C" void FnecP_rawAdd(FnecRawElement pRawResult, const FnecRawElement pRawA, const FnecRawElement pRawB);
    extern "C" void FnecP_rawSub(FnecRawElement pRawResult, const FnecRawElement pRawA, const FnecRawElement pRawB);
    extern "C" void FnecP_rawNeg(FnecRawElement pRawResult, const FnecRawElement pRawA);
    extern "C" void FnecP_rawMMul(FnecRawElement pRawResult, const FnecRawElement pRawA, const FnecRawElement pRawB);
    extern "C" void FnecP_rawMSquare(FnecRawElement pRawResult, const FnecRawElement pRawA);
    extern "C" void FnecP_rawMMul1(FnecRawElement pRawResult, const FnecRawElement pRawA, uint64_t pRawB);
    extern "C" void FnecP_rawToMontgomery(FnecRawElement pRawResult, const FnecRawElement &pRawA);
    extern "C" void FnecP_rawFromMontgomery(FnecRawElement pRawResult, const FnecRawElement &pRawA);
    extern "C" int FnecP_rawIsEq(const FnecRawElement pRawA, const FnecRawElement pRawB);
    extern "C" int FnecP_rawIsZero(const FnecRawElement pRawB);

    extern "C" void FnecP_fail();

#else

    static FnecElement FnecP_q;
    static FnecElement FnecP_R3;
    static FnecRawElement FnecP_rawq;
    static FnecRawElement FnecP_rawR3;

    inline void FnecP_copy(PFnecElement r, PFnecElement a){
        std::cerr << "FnecP_copy() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_copyn(PFnecElement r, PFnecElement a, int n){
        std::cerr << "FnecP_copyn() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_add(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_add() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_sub(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_sub() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_neg(PFnecElement r, PFnecElement a) {
        std::cerr << "FnecP_neg() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_mul(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_mul() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_square(PFnecElement r, PFnecElement a) {
        std::cerr << "FnecP_square() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_band(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_band() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_bor(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_bor() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_bxor(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_bxor() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_bnot(PFnecElement r, PFnecElement a) {
        std::cerr << "FnecP_bnot() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_shl(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_shl() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_shr(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_shr() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_eq(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_eq() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_neq(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_neq() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_lt(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_lt() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_gt(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_gt() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_leq(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_leq() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_geq(PFnecElement r, PFnecElement a, PFnecElement b){
        std::cerr << "FnecP_geq() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_land(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_land() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_lor(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "FnecP_lor() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_lnot(PFnecElement r, PFnecElement a) {
        std::cerr << "FnecP_lnot() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_toNormal(PFnecElement r, PFnecElement a) {
        std::cerr << "FnecP_toNormal() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_toLongNormal(PFnecElement r, PFnecElement a) {
        std::cerr << "FnecP_toLongNormal() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_toMontgomery(PFnecElement r, PFnecElement a) {
        std::cerr << "FnecP_toMontgomery() not implemented in C++ code." << std::endl;
        assert(true);
    }

    inline int FnecP_isTrue(PFnecElement pE){
        std::cerr << "FnecP_isTrue() not implemented in C++ code." << std::endl;
        assert(true);
        return 0; // Placeholder return value
    }
    inline int FnecP_toInt(PFnecElement pE) {
        std::cerr << "FnecP_toInt() not implemented in C++ code." << std::endl;
        assert(true);
        return 0; // Placeholder return value
    }

    inline void FnecP_rawCopy(FnecRawElement pRawResult, const FnecRawElement pRawA) {
        std::cerr << "FnecP_rawCopy() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_rawSwap(FnecRawElement pRawResult, FnecRawElement pRawA) {
        std::cerr << "FnecP_rawSwap() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_rawAdd(FnecRawElement pRawResult, const FnecRawElement pRawA, const FnecRawElement pRawB) {
        std::cerr << "FnecP_rawAdd() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_rawSub(FnecRawElement pRawResult, const FnecRawElement pRawA, const FnecRawElement pRawB) {
        std::cerr << "FnecP_rawSub() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_rawNeg(FnecRawElement pRawResult, const FnecRawElement pRawA) {
        std::cerr << "FnecP_rawNeg() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_rawMMul(FnecRawElement pRawResult, const FnecRawElement pRawA, const FnecRawElement pRawB) {
        std::cerr << "FnecP_rawMMul() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_rawMSquare(FnecRawElement pRawResult, const FnecRawElement pRawA) {
        std::cerr << "FnecP_rawMSquare() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_rawMMul1(FnecRawElement pRawResult, const FnecRawElement pRawA, uint64_t pRawB) {
        std::cerr << "FnecP_rawMMul1() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline void FnecP_rawToMontgomery(FnecRawElement pRawResult, const FnecRawElement &pRawA) {
        std::cerr << "FnecP_rawToMontgomery() not implemented in C++ code." << std::endl;
        assert(false);
    }
    inline void FnecP_rawFromMontgomery(FnecRawElement pRawResult, const FnecRawElement &pRawA) {
        std::cerr << "FnecP_rawFromMontgomery() not implemented in C++ code." << std::endl;
        assert(true);
    }
    inline int FnecP_rawIsEq(const FnecRawElement pRawA, const FnecRawElement pRawB) {
        std::cerr << "FnecP_rawIsEq() not implemented in C++ code." << std::endl;
        assert(true);
        return 0;
    }
    inline int FnecP_rawIsZero(const FnecRawElement pRawB) {
        std::cerr << "FnecP_rawIsZero() not implemented in C++ code." << std::endl;
        assert(true);
        return 0;
    }
    inline void FnecP_fail() {
        assert(true);
    }
#endif



// Pending functions to convert

void FnecP_str2element(PFnecElement pE, char const*s);
char *FnecP_element2str(PFnecElement pE);
void FnecP_idiv(PFnecElement r, PFnecElement a, PFnecElement b);
void FnecP_mod(PFnecElement r, PFnecElement a, PFnecElement b);
void FnecP_inv(PFnecElement r, PFnecElement a);
void FnecP_div(PFnecElement r, PFnecElement a, PFnecElement b);
void FnecP_pow(PFnecElement r, PFnecElement a, PFnecElement b);

class RawFnecP {

public:
    const static int N64 = FnecP_N64;
    const static int MaxBits = 256;


    struct Element {
        FnecRawElement v;
    };

private:
    Element fZero;
    Element fOne;
    Element fNegOne;

public:

    RawFnecP();
    ~RawFnecP();

    const Element &zero() { return fZero; };
    const Element &one() { return fOne; };
    const Element &negOne() { return fNegOne; };
    Element set(int value);
    void set(Element &r, int value);

    void fromString(Element &r, const std::string &n, uint32_t radix = 10);
    std::string toString(const Element &a, uint32_t radix = 10);

    void inline copy(Element &r, const Element &a) { FnecP_rawCopy(r.v, a.v); };
    void inline swap(Element &a, Element &b) { FnecP_rawSwap(a.v, b.v); };
    void inline add(Element &r, const Element &a, const Element &b) { FnecP_rawAdd(r.v, a.v, b.v); };
    void inline sub(Element &r, const Element &a, const Element &b) { FnecP_rawSub(r.v, a.v, b.v); };
    void inline mul(Element &r, const Element &a, const Element &b) { FnecP_rawMMul(r.v, a.v, b.v); };

    Element inline add(const Element &a, const Element &b) { Element r; FnecP_rawAdd(r.v, a.v, b.v); return r;};
    Element inline sub(const Element &a, const Element &b) { Element r; FnecP_rawSub(r.v, a.v, b.v); return r;};
    Element inline mul(const Element &a, const Element &b) { Element r; FnecP_rawMMul(r.v, a.v, b.v); return r;};

    Element inline neg(const Element &a) { Element r; FnecP_rawNeg(r.v, a.v); return r; };
    Element inline square(const Element &a) { Element r; FnecP_rawMSquare(r.v, a.v); return r; };

    Element inline add(int a, const Element &b) { return add(set(a), b);};
    Element inline sub(int a, const Element &b) { return sub(set(a), b);};
    Element inline mul(int a, const Element &b) { return mul(set(a), b);};

    Element inline add(const Element &a, int b) { return add(a, set(b));};
    Element inline sub(const Element &a, int b) { return sub(a, set(b));};
    Element inline mul(const Element &a, int b) { return mul(a, set(b));};
    
    void inline mul1(Element &r, const Element &a, uint64_t b) { FnecP_rawMMul1(r.v, a.v, b); };
    void inline neg(Element &r, const Element &a) { FnecP_rawNeg(r.v, a.v); };
    void inline square(Element &r, const Element &a) { FnecP_rawMSquare(r.v, a.v); };
    void inv(Element &r, const Element &a);
    void div(Element &r, const Element &a, const Element &b);
    void exp(Element &r, const Element &base, uint8_t* scalar, unsigned int scalarSize);

    void inline toMontgomery(Element &r, const Element &a) { FnecP_rawToMontgomery(r.v, a.v); };
    void inline fromMontgomery(Element &r, const Element &a) { FnecP_rawFromMontgomery(r.v, a.v); };
    int inline eq(const Element &a, const Element &b) { return FnecP_rawIsEq(a.v, b.v); };
    int inline isZero(const Element &a) { return FnecP_rawIsZero(a.v); };

    void toMpz(mpz_t r, const Element &a);
    void fromMpz(Element &a, const mpz_t r);

    int toRprBE(const Element &element, uint8_t *data, int bytes);
    int fromRprBE(Element &element, const uint8_t *data, int bytes);
    
    int bytes ( void ) { return FnecP_N64 * 8; };
    
    void fromUI(Element &r, unsigned long int v);

    static RawFnecP field;

};


#endif // __FNEC_P_H



