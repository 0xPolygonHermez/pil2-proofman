#ifndef CMUL_GOLDILOCKS
#define CMUL_GOLDILOCKS

#include "goldilocks_base_field.hpp"

void CMul(uint64_t* out, uint *size_out, uint64_t *ina, uint* size_ina, uint64_t *inb, uint *size_inb)
{
    Goldilocks::Element *a = (Goldilocks::Element *)ina;
    Goldilocks::Element *b = (Goldilocks::Element *)inb;
    Goldilocks::Element A = (a[0] + a[1]) * (b[0] + b[1]);
    Goldilocks::Element B = (a[0] + a[2]) * (b[0] + b[2]);
    Goldilocks::Element C = (a[1] + a[2]) * (b[1] + b[2]);
    Goldilocks::Element D = a[0] * b[0];
    Goldilocks::Element E = a[1] * b[1];
    Goldilocks::Element F = a[2] * b[2];
    Goldilocks::Element G = D - E;

    out[0] = Goldilocks::toU64((C + G) - F);
    out[1] = Goldilocks::toU64(((((A + C) - E) - E) - D));
    out[2] = Goldilocks::toU64(B - G);
}

void CMulAdd(Goldilocks::Element *out, Goldilocks::Element *a, Goldilocks::Element *b, Goldilocks::Element *c) {
    Goldilocks::Element A = (a[0] + a[1]) * (b[0] + b[1]);
    Goldilocks::Element B = (a[0] + a[2]) * (b[0] + b[2]);
    Goldilocks::Element C = (a[1] + a[2]) * (b[1] + b[2]);
    Goldilocks::Element D = a[0] * b[0];
    Goldilocks::Element E = a[1] * b[1];
    Goldilocks::Element F = a[2] * b[2];
    Goldilocks::Element G = D - E;

    out[0] = (C + G) - F + c[0];
    out[1] = (((A + C) - E) - E) - D + c[1];
    out[2] = B - G + c[2];
}

// Outputs first, in declaration order -- circom's extern_c convention. Adding one changes this
// symbol's mangled name, so the generated verifier.cpp and this file move together.
void EvPol4(uint64_t* out, uint *size_out, uint64_t *s, uint *size_s, uint64_t *estrin, uint *size_estrin,
            uint64_t *coefs, uint* size_coefs, uint64_t *x, uint *size_x)
{
    Goldilocks::Element coefs_[5][3];
    uint64_t c = 0;
    for(uint64_t i = 0; i < 5; ++i) {
        for(uint64_t j = 0; j < 3; ++j) {
            coefs_[i][j] = Goldilocks::fromU64(coefs[c++]);
        }
    }

    // One Estrin chain, not Horner plus Estrin: `out` falls out of the same five products.
    //   s = x*x    estrin = c4*s + (c3*x + c2)    out = estrin*s + (c1*x + c0)
    Goldilocks::Element *x_ = (Goldilocks::Element *)x;
    Goldilocks::Element zero[3] = {Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero()};
    Goldilocks::Element s_[3], hi[3], t[3], o[3];
    CMulAdd(s_, x_, x_, zero);
    CMulAdd(t, coefs_[3], x_, coefs_[2]);
    CMulAdd(hi, coefs_[4], s_, t);
    CMulAdd(t, coefs_[1], x_, coefs_[0]);
    CMulAdd(o, hi, s_, t);
    for(uint64_t i = 0; i < 3; ++i) {
        out[i] = Goldilocks::toU64(o[i]);
        s[i] = Goldilocks::toU64(s_[i]);
        estrin[i] = Goldilocks::toU64(hi[i]);
    }
}


#endif