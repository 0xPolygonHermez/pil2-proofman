#include "fnec.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <assert.h>
#include <string>


static mpz_t q;
static mpz_t zero;
static mpz_t one;
static mpz_t mask;
static size_t nBits;
static bool initialized = false;


void FnecP_toMpz(mpz_t r, PFnecElement pE) {
    FnecElement tmp;
    FnecP_toNormal(&tmp, pE);
    if (!(tmp.type & FnecP_LONG)) {
        mpz_set_si(r, tmp.shortVal);
        if (tmp.shortVal<0) {
            mpz_add(r, r, q);
        }
    } else {
        mpz_import(r, FnecP_N64, -1, 8, -1, 0, (const void *)tmp.longVal);
    }
}

void FnecP_fromMpz(PFnecElement pE, mpz_t v) {
    if (mpz_fits_sint_p(v)) {
        pE->type = FnecP_SHORT;
        pE->shortVal = mpz_get_si(v);
    } else {
        pE->type = FnecP_LONG;
        for (int i=0; i<FnecP_N64; i++) pE->longVal[i] = 0;
        mpz_export((void *)(pE->longVal), NULL, -1, 8, -1, 0, v);
    }
}


bool FnecP_init() {
    if (initialized) return false;
    initialized = true;
    mpz_init(q);
    mpz_import(q, FnecP_N64, -1, 8, -1, 0, (const void *)FnecP_q.longVal);
    mpz_init_set_ui(zero, 0);
    mpz_init_set_ui(one, 1);
    nBits = mpz_sizeinbase (q, 2);
    mpz_init(mask);
    mpz_mul_2exp(mask, one, nBits);
    mpz_sub(mask, mask, one);
    return true;
}

void FnecP_str2element(PFnecElement pE, char const *s) {
    mpz_t mr;
    mpz_init_set_str(mr, s, 10);
    mpz_fdiv_r(mr, mr, q);
    FnecP_fromMpz(pE, mr);
    mpz_clear(mr);
}

char *FnecP_element2str(PFnecElement pE) {
    FnecElement tmp;
    mpz_t r;
    if (!(pE->type & FnecP_LONG)) {
        if (pE->shortVal>=0) {
            char *r = new char[32];
            snprintf(r, 32, "%d", pE->shortVal);
            return r;
        } else {
            mpz_init_set_si(r, pE->shortVal);
            mpz_add(r, r, q);
        }
    } else {
        FnecP_toNormal(&tmp, pE);
        mpz_init(r);
        mpz_import(r, FnecP_N64, -1, 8, -1, 0, (const void *)tmp.longVal);
    }
    char *res = mpz_get_str (0, 10, r);
    mpz_clear(r);
    return res;
}

void FnecP_idiv(PFnecElement r, PFnecElement a, PFnecElement b) {
    mpz_t ma;
    mpz_t mb;
    mpz_t mr;
    mpz_init(ma);
    mpz_init(mb);
    mpz_init(mr);

    FnecP_toMpz(ma, a);
    // char *s1 = mpz_get_str (0, 10, ma);
    // printf("s1 %s\n", s1);
    FnecP_toMpz(mb, b);
    // char *s2 = mpz_get_str (0, 10, mb);
    // printf("s2 %s\n", s2);
    mpz_fdiv_q(mr, ma, mb);
    // char *sr = mpz_get_str (0, 10, mr);
    // printf("r %s\n", sr);
    FnecP_fromMpz(r, mr);

    mpz_clear(ma);
    mpz_clear(mb);
    mpz_clear(mr);
}

void FnecP_mod(PFnecElement r, PFnecElement a, PFnecElement b) {
    mpz_t ma;
    mpz_t mb;
    mpz_t mr;
    mpz_init(ma);
    mpz_init(mb);
    mpz_init(mr);

    FnecP_toMpz(ma, a);
    FnecP_toMpz(mb, b);
    mpz_fdiv_r(mr, ma, mb);
    FnecP_fromMpz(r, mr);

    mpz_clear(ma);
    mpz_clear(mb);
    mpz_clear(mr);
}

void FnecP_pow(PFnecElement r, PFnecElement a, PFnecElement b) {
    mpz_t ma;
    mpz_t mb;
    mpz_t mr;
    mpz_init(ma);
    mpz_init(mb);
    mpz_init(mr);

    FnecP_toMpz(ma, a);
    FnecP_toMpz(mb, b);
    mpz_powm(mr, ma, mb, q);
    FnecP_fromMpz(r, mr);

    mpz_clear(ma);
    mpz_clear(mb);
    mpz_clear(mr);
}

void FnecP_inv(PFnecElement r, PFnecElement a) {
    mpz_t ma;
    mpz_t mr;
    mpz_init(ma);
    mpz_init(mr);

    FnecP_toMpz(ma, a);
    mpz_invert(mr, ma, q);
    FnecP_fromMpz(r, mr);
    mpz_clear(ma);
    mpz_clear(mr);
}

void FnecP_div(PFnecElement r, PFnecElement a, PFnecElement b) {
    FnecElement tmp;
    FnecP_inv(&tmp, b);
    FnecP_mul(r, a, &tmp);
}

#ifdef __USE_ASSEMBLY__
void FnecP_fail() {
    assert(false);
}
#endif

RawFnecP::RawFnecP() {
#ifdef __USE_ASSEMBLY__
    FnecP_init();
    set(fZero, 0);
    set(fOne, 1);
    neg(fNegOne, fOne);
#endif
}

RawFnecP::~RawFnecP() {
}

void RawFnecP::fromString(Element &r, const std::string &s, uint32_t radix) {
    mpz_t mr;
    mpz_init_set_str(mr, s.c_str(), radix);
    mpz_fdiv_r(mr, mr, q);
    for (int i=0; i<FnecP_N64; i++) r.v[i] = 0;
    mpz_export((void *)(r.v), NULL, -1, 8, -1, 0, mr);
    FnecP_rawToMontgomery(r.v,r.v);
    mpz_clear(mr);
}

void RawFnecP::fromUI(Element &r, unsigned long int v) {
    mpz_t mr;
    mpz_init(mr);
    mpz_set_ui(mr, v);
    for (int i=0; i<FnecP_N64; i++) r.v[i] = 0;
    mpz_export((void *)(r.v), NULL, -1, 8, -1, 0, mr);
    FnecP_rawToMontgomery(r.v,r.v);
    mpz_clear(mr);
}

RawFnecP::Element RawFnecP::set(int value) {
  Element r;
  set(r, value);
  return r;
}

void RawFnecP::set(Element &r, int value) {
  mpz_t mr;
  mpz_init(mr);
  mpz_set_si(mr, value);
  if (value < 0) {
      mpz_add(mr, mr, q);
  }

  mpz_export((void *)(r.v), NULL, -1, 8, -1, 0, mr);
      
  for (int i=0; i<FnecP_N64; i++) r.v[i] = 0;
  mpz_export((void *)(r.v), NULL, -1, 8, -1, 0, mr);
  FnecP_rawToMontgomery(r.v,r.v);
  mpz_clear(mr);
}

std::string RawFnecP::toString(const Element &a, uint32_t radix) {
    Element tmp;
    mpz_t r;
    FnecP_rawFromMontgomery(tmp.v, a.v);
    mpz_init(r);
    mpz_import(r, FnecP_N64, -1, 8, -1, 0, (const void *)(tmp.v));
    char *res = mpz_get_str (0, radix, r);
    mpz_clear(r);
    std::string resS(res);
    free(res);
    return resS;
}

void RawFnecP::inv(Element &r, const Element &a) {
    mpz_t mr;
    mpz_init(mr);
    mpz_import(mr, FnecP_N64, -1, 8, -1, 0, (const void *)(a.v));
    mpz_invert(mr, mr, q);


    for (int i=0; i<FnecP_N64; i++) r.v[i] = 0;
    mpz_export((void *)(r.v), NULL, -1, 8, -1, 0, mr);

    FnecP_rawMMul(r.v, r.v,FnecP_rawR3);
    mpz_clear(mr);
}

void RawFnecP::div(Element &r, const Element &a, const Element &b) {
    Element tmp;
    inv(tmp, b);
    mul(r, a, tmp);
}

#define BIT_IS_SET(s, p) (s[p>>3] & (1 << (p & 0x7)))
void RawFnecP::exp(Element &r, const Element &base, uint8_t* scalar, unsigned int scalarSize) {
    bool oneFound = false;
    Element copyBase;
    copy(copyBase, base);
    for (int i=scalarSize*8-1; i>=0; i--) {
        if (!oneFound) {
            if ( !BIT_IS_SET(scalar, i) ) continue;
            copy(r, copyBase);
            oneFound = true;
            continue;
        }
        square(r, r);
        if ( BIT_IS_SET(scalar, i) ) {
            mul(r, r, copyBase);
        }
    }
    if (!oneFound) {
        copy(r, fOne);
    }
}

void RawFnecP::toMpz(mpz_t r, const Element &a) {
    Element tmp;
    FnecP_rawFromMontgomery(tmp.v, a.v);
    mpz_import(r, FnecP_N64, -1, 8, -1, 0, (const void *)tmp.v);
}

void RawFnecP::fromMpz(Element &r, const mpz_t a) {
    for (int i=0; i<FnecP_N64; i++) r.v[i] = 0;
    mpz_export((void *)(r.v), NULL, -1, 8, -1, 0, a);
    FnecP_rawToMontgomery(r.v, r.v);
}

int RawFnecP::toRprBE(const Element &element, uint8_t *data, int bytes)
{
    if (bytes < FnecP_N64 * 8) {
      return -(FnecP_N64 * 8);
    }

    mpz_t r;
    mpz_init(r);
  
    toMpz(r, element);
    
    mpz_export(data, NULL, 1, 8, 1, 0, r);
  
    mpz_clear(r);
    return FnecP_N64 * 8;
}

int RawFnecP::fromRprBE(Element &element, const uint8_t *data, int bytes)
{
    if (bytes < FnecP_N64 * 8) {
      return -(FnecP_N64* 8);
    }
    mpz_t r;
    mpz_init(r);

    mpz_import(r, FnecP_N64 * 8, 0, 1, 0, 0, data);
    fromMpz(element, r);

    mpz_clear(r);
    return FnecP_N64 * 8;
}

static bool init = FnecP_init();

RawFnecP RawFnecP::field;

