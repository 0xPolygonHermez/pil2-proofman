#ifndef __FNEC_H
#define __FNEC_H

#include <stdint.h>
#include <string>
#include <gmp.h>
#include <iostream>
#include <cassert>

#define Fnec_N64 4
#define Fnec_SHORT 0x00000000
#define Fnec_LONG 0x80000000
#define Fnec_LONGMONTGOMERY 0xC0000000
typedef uint64_t FnecRawElement[Fnec_N64];
typedef struct __attribute__((__packed__)) {
    int32_t shortVal;
    uint32_t type;
    FnecRawElement longVal;
} FnecElement;
typedef FnecElement *PFnecElement;

#ifdef __USE_ASSEMBLY__
    extern FnecElement Fnec_q;
    extern FnecElement Fnec_R3;
    extern FnecRawElement Fnec_rawq;
    extern FnecRawElement Fnec_rawR3;

    extern "C" void Fnec_copy(PFnecElement r, PFnecElement a);
    extern "C" void Fnec_copyn(PFnecElement r, PFnecElement a, int n);
    extern "C" void Fnec_add(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_sub(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_neg(PFnecElement r, PFnecElement a);
    extern "C" void Fnec_mul(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_square(PFnecElement r, PFnecElement a);
    extern "C" void Fnec_band(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_bor(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_bxor(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_bnot(PFnecElement r, PFnecElement a);
    extern "C" void Fnec_shl(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_shr(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_eq(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_neq(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_lt(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_gt(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_leq(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_geq(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_land(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_lor(PFnecElement r, PFnecElement a, PFnecElement b);
    extern "C" void Fnec_lnot(PFnecElement r, PFnecElement a);
    extern "C" void Fnec_toNormal(PFnecElement r, PFnecElement a);
    extern "C" void Fnec_toLongNormal(PFnecElement r, PFnecElement a);
    extern "C" void Fnec_toMontgomery(PFnecElement r, PFnecElement a);

    extern "C" int Fnec_isTrue(PFnecElement pE);
    extern "C" int Fnec_toInt(PFnecElement pE);

    extern "C" void Fnec_rawCopy(FnecRawElement pRawResult, const FnecRawElement pRawA);
    extern "C" void Fnec_rawSwap(FnecRawElement pRawResult, FnecRawElement pRawA);
    extern "C" void Fnec_rawAdd(FnecRawElement pRawResult, const FnecRawElement pRawA, const FnecRawElement pRawB);
    extern "C" void Fnec_rawSub(FnecRawElement pRawResult, const FnecRawElement pRawA, const FnecRawElement pRawB);
    extern "C" void Fnec_rawNeg(FnecRawElement pRawResult, const FnecRawElement pRawA);
    extern "C" void Fnec_rawMMul(FnecRawElement pRawResult, const FnecRawElement pRawA, const FnecRawElement pRawB);
    extern "C" void Fnec_rawMSquare(FnecRawElement pRawResult, const FnecRawElement pRawA);
    extern "C" void Fnec_rawMMul1(FnecRawElement pRawResult, const FnecRawElement pRawA, uint64_t pRawB);
    extern "C" void Fnec_rawToMontgomery(FnecRawElement pRawResult, const FnecRawElement &pRawA);
    extern "C" void Fnec_rawFromMontgomery(FnecRawElement pRawResult, const FnecRawElement &pRawA);
    extern "C" int Fnec_rawIsEq(const FnecRawElement pRawA, const FnecRawElement pRawB);
    extern "C" int Fnec_rawIsZero(const FnecRawElement pRawB);

    extern "C" void Fnec_fail();

#else

    FnecElement Fnec_q;
    FnecElement Fnec_R3;
    FnecRawElement Fnec_rawq;
    FnecRawElement Fnec_rawR3;

    void Fnec_copy(PFnecElement r, PFnecElement a){
        std::cerr << "Fnec_copy() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_copyn(PFnecElement r, PFnecElement a, int n){
        std::cerr << "Fnec_copyn() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_add(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_add() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_sub(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_sub() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_neg(PFnecElement r, PFnecElement a) {
        std::cerr << "Fnec_neg() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_mul(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_mul() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_square(PFnecElement r, PFnecElement a) {
        std::cerr << "Fnec_square() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_band(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_band() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_bor(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_bor() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_bxor(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_bxor() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_bnot(PFnecElement r, PFnecElement a) {
        std::cerr << "Fnec_bnot() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_shl(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_shl() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_shr(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_shr() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_eq(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_eq() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_neq(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_neq() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_lt(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_lt() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_gt(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_gt() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_leq(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_leq() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_geq(PFnecElement r, PFnecElement a, PFnecElement b){
        std::cerr << "Fnec_geq() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_land(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_land() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_lor(PFnecElement r, PFnecElement a, PFnecElement b) {
        std::cerr << "Fnec_lor() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_lnot(PFnecElement r, PFnecElement a) {
        std::cerr << "Fnec_lnot() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_toNormal(PFnecElement r, PFnecElement a) {
        std::cerr << "Fnec_toNormal() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_toLongNormal(PFnecElement r, PFnecElement a) {
        std::cerr << "Fnec_toLongNormal() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_toMontgomery(PFnecElement r, PFnecElement a) {
        std::cerr << "Fnec_toMontgomery() not implemented in C++ code." << std::endl;
        assert(false);
    }

    int Fnec_isTrue(PFnecElement pE){
        std::cerr << "Fnec_isTrue() not implemented in C++ code." << std::endl;
        assert(false);
        return 0; // Placeholder return value
    }
    int Fnec_toInt(PFnecElement pE) {
        std::cerr << "Fnec_toInt() not implemented in C++ code." << std::endl;
        assert(false);
        return 0; // Placeholder return value
    }

    void Fnec_rawCopy(FnecRawElement pRawResult, const FnecRawElement pRawA) {
        std::cerr << "Fnec_rawCopy() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_rawSwap(FnecRawElement pRawResult, FnecRawElement pRawA) {
        std::cerr << "Fnec_rawSwap() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_rawAdd(FnecRawElement pRawResult, const FnecRawElement pRawA, const FnecRawElement pRawB) {
        std::cerr << "Fnec_rawAdd() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_rawSub(FnecRawElement pRawResult, const FnecRawElement pRawA, const FnecRawElement pRawB) {
        std::cerr << "Fnec_rawSub() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_rawNeg(FnecRawElement pRawResult, const FnecRawElement pRawA) {
        std::cerr << "Fnec_rawNeg() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_rawMMul(FnecRawElement pRawResult, const FnecRawElement pRawA, const FnecRawElement pRawB) {
        std::cerr << "Fnec_rawMMul() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_rawMSquare(FnecRawElement pRawResult, const FnecRawElement pRawA) {
        std::cerr << "Fnec_rawMSquare() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_rawMMul1(FnecRawElement pRawResult, const FnecRawElement pRawA, uint64_t pRawB) {
        std::cerr << "Fnec_rawMMul1() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_rawToMontgomery(FnecRawElement pRawResult, const FnecRawElement &pRawA) {
        std::cerr << "Fnec_rawToMontgomery() not implemented in C++ code." << std::endl;
        assert(false);
    }
    void Fnec_rawFromMontgomery(FnecRawElement pRawResult, const FnecRawElement &pRawA) {
        std::cerr << "Fnec_rawFromMontgomery() not implemented in C++ code." << std::endl;
        assert(false);
    }
    int Fnec_rawIsEq(const FnecRawElement pRawA, const FnecRawElement pRawB) {
        std::cerr << "Fnec_rawIsEq() not implemented in C++ code." << std::endl;
        assert(false);
        return 0;
    }
    int Fnec_rawIsZero(const FnecRawElement pRawB) {
        std::cerr << "Fnec_rawIsZero() not implemented in C++ code." << std::endl;
        assert(false);
        return 0;
    }
#endif
void Fnec_fail() {
    assert(false);
}


// Pending functions to convert

void Fnec_str2element(PFnecElement pE, char const*s);
char *Fnec_element2str(PFnecElement pE);
void Fnec_idiv(PFnecElement r, PFnecElement a, PFnecElement b);
void Fnec_mod(PFnecElement r, PFnecElement a, PFnecElement b);
void Fnec_inv(PFnecElement r, PFnecElement a);
void Fnec_div(PFnecElement r, PFnecElement a, PFnecElement b);
void Fnec_pow(PFnecElement r, PFnecElement a, PFnecElement b);

class RawFnec {

public:
    const static int N64 = Fnec_N64;
    const static int MaxBits = 256;


    struct Element {
        FnecRawElement v;
    };

private:
    Element fZero;
    Element fOne;
    Element fNegOne;

public:

    RawFnec();
    ~RawFnec();

    const Element &zero() { return fZero; };
    const Element &one() { return fOne; };
    const Element &negOne() { return fNegOne; };
    Element set(int value);
    void set(Element &r, int value);

    void fromString(Element &r, const std::string &n, uint32_t radix = 10);
    std::string toString(const Element &a, uint32_t radix = 10);

    void inline copy(Element &r, const Element &a) { Fnec_rawCopy(r.v, a.v); };
    void inline swap(Element &a, Element &b) { Fnec_rawSwap(a.v, b.v); };
    void inline add(Element &r, const Element &a, const Element &b) { Fnec_rawAdd(r.v, a.v, b.v); };
    void inline sub(Element &r, const Element &a, const Element &b) { Fnec_rawSub(r.v, a.v, b.v); };
    void inline mul(Element &r, const Element &a, const Element &b) { Fnec_rawMMul(r.v, a.v, b.v); };

    Element inline add(const Element &a, const Element &b) { Element r; Fnec_rawAdd(r.v, a.v, b.v); return r;};
    Element inline sub(const Element &a, const Element &b) { Element r; Fnec_rawSub(r.v, a.v, b.v); return r;};
    Element inline mul(const Element &a, const Element &b) { Element r; Fnec_rawMMul(r.v, a.v, b.v); return r;};

    Element inline neg(const Element &a) { Element r; Fnec_rawNeg(r.v, a.v); return r; };
    Element inline square(const Element &a) { Element r; Fnec_rawMSquare(r.v, a.v); return r; };

    Element inline add(int a, const Element &b) { return add(set(a), b);};
    Element inline sub(int a, const Element &b) { return sub(set(a), b);};
    Element inline mul(int a, const Element &b) { return mul(set(a), b);};

    Element inline add(const Element &a, int b) { return add(a, set(b));};
    Element inline sub(const Element &a, int b) { return sub(a, set(b));};
    Element inline mul(const Element &a, int b) { return mul(a, set(b));};
    
    void inline mul1(Element &r, const Element &a, uint64_t b) { Fnec_rawMMul1(r.v, a.v, b); };
    void inline neg(Element &r, const Element &a) { Fnec_rawNeg(r.v, a.v); };
    void inline square(Element &r, const Element &a) { Fnec_rawMSquare(r.v, a.v); };
    void inv(Element &r, const Element &a);
    void div(Element &r, const Element &a, const Element &b);
    void exp(Element &r, const Element &base, uint8_t* scalar, unsigned int scalarSize);

    void inline toMontgomery(Element &r, const Element &a) { Fnec_rawToMontgomery(r.v, a.v); };
    void inline fromMontgomery(Element &r, const Element &a) { Fnec_rawFromMontgomery(r.v, a.v); };
    int inline eq(const Element &a, const Element &b) { return Fnec_rawIsEq(a.v, b.v); };
    int inline isZero(const Element &a) { return Fnec_rawIsZero(a.v); };

    void toMpz(mpz_t r, const Element &a);
    void fromMpz(Element &a, const mpz_t r);

    int toRprBE(const Element &element, uint8_t *data, int bytes);
    int fromRprBE(Element &element, const uint8_t *data, int bytes);
    
    int bytes ( void ) { return Fnec_N64 * 8; };
    
    void fromUI(Element &r, unsigned long int v);

    static RawFnec field;

};


#endif // __FNEC_H



