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

void EvPol4(uint64_t* out, uint *size_out, uint64_t *coefs, uint* size_coefs, uint64_t *x, uint *size_x)
{
    Goldilocks::Element coefs_[5][3];
    uint64_t c = 0;
    for(uint64_t i = 0; i < 5; ++i) {
        for(uint64_t j = 0; j < 3; ++j) {
            coefs_[i][j] = Goldilocks::fromU64(coefs[c++]);
        }
    }

    Goldilocks::Element *x_ = (Goldilocks::Element *)x;
    Goldilocks::Element acc[3];
    CMulAdd(acc, coefs_[4], x_, coefs_[3]);
    CMulAdd(acc, acc, x_, coefs_[2]);
    CMulAdd(acc, acc, x_, coefs_[1]);
    CMulAdd(acc, acc, x_, coefs_[0]);
    out[0] = Goldilocks::toU64(acc[0]);
    out[1] = Goldilocks::toU64(acc[1]);
    out[2] = Goldilocks::toU64(acc[2]);
}


#endif