#ifndef GOLDILOCKS_CUBIC_EXTENSION_PACK
#define GOLDILOCKS_CUBIC_EXTENSION_PACK
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include <cassert>
/*
    Implementations for expressions:
*/

inline void Goldilocks3::copy_pack( uint64_t nrowsPack, Goldilocks::Element *c_, const Goldilocks::Element *a_, const bool const_a)
{
    if(const_a) {
        for(uint64_t irow =0; irow<nrowsPack; ++irow){
            Goldilocks::copy(c_[irow], a_[0]);
            Goldilocks::copy(c_[nrowsPack + irow], a_[1]);
            Goldilocks::copy(c_[2*nrowsPack + irow], a_[2]);   
        }
    } else {
        for(uint64_t irow =0; irow<nrowsPack; ++irow){
            Goldilocks::copy(c_[irow], a_[irow]);
            Goldilocks::copy(c_[nrowsPack + irow], a_[nrowsPack + irow]);
            Goldilocks::copy(c_[2*nrowsPack + irow], a_[2*nrowsPack + irow]);   
        }
    }
}

inline void Goldilocks3::copy_pack( uint64_t nrowsPack, Goldilocks::Element *c_, const Goldilocks::Element *a_)
{
    for(uint64_t irow =0; irow<nrowsPack; ++irow){
        Goldilocks::copy(c_[irow], a_[irow]);
        Goldilocks::copy(c_[nrowsPack + irow], a_[nrowsPack + irow]);
        Goldilocks::copy(c_[2*nrowsPack + irow], a_[2*nrowsPack + irow]);   
    }
}

inline void Goldilocks3::add_pack( uint64_t nrowsPack, Goldilocks::Element *c_, const Goldilocks::Element *a_, const bool const_a, const Goldilocks::Element *b_, const bool const_b)
{
    if(const_a && const_b) {
        for(uint64_t irow =0; irow<nrowsPack; ++irow){
            c_[irow] = a_[0] + b_[0];
            c_[nrowsPack + irow] = a_[1] + b_[1];
            c_[2*nrowsPack + irow] = a_[2] + b_[2];
        }
    } else if(const_a) {
        for(uint64_t irow =0; irow<nrowsPack; ++irow){
            c_[irow] = a_[0] + b_[irow];
            c_[nrowsPack + irow] = a_[1] + b_[nrowsPack + irow];
            c_[2*nrowsPack + irow] = a_[2] + b_[2*nrowsPack + irow];
        }
    } else if(const_b) {
        for(uint64_t irow =0; irow<nrowsPack; ++irow){
            c_[irow] = a_[irow] + b_[0];
            c_[nrowsPack + irow] = a_[nrowsPack + irow] + b_[1];
            c_[2*nrowsPack + irow] = a_[2*nrowsPack + irow] + b_[2];
        }
    } else {
        for(uint64_t irow =0; irow<nrowsPack; ++irow){
            c_[irow] = a_[irow] + b_[irow];
            c_[nrowsPack + irow] = a_[nrowsPack + irow] + b_[nrowsPack + irow];
            c_[2*nrowsPack + irow] = a_[2*nrowsPack + irow] + b_[2*nrowsPack + irow];
        }
    }
    
}

inline void Goldilocks3::sub_pack( uint64_t nrowsPack, Goldilocks::Element *c_, const Goldilocks::Element *a_, const bool const_a, const Goldilocks::Element *b_, const bool const_b)
{
    if(const_a && const_b) {
        for(uint64_t irow =0; irow<nrowsPack; ++irow){
            c_[irow] = a_[0] - b_[0];
            c_[nrowsPack + irow] = a_[1] - b_[1];
            c_[2*nrowsPack + irow] = a_[2] - b_[2];
        }
    } else if(const_a) {
        for(uint64_t irow =0; irow<nrowsPack; ++irow){
            c_[irow] = a_[0] - b_[irow];
            c_[nrowsPack + irow] = a_[1] - b_[nrowsPack + irow];
            c_[2*nrowsPack + irow] = a_[2] - b_[2*nrowsPack + irow];
        }
    } else if(const_b) {
        for(uint64_t irow =0; irow<nrowsPack; ++irow){
            c_[irow] = a_[irow] - b_[0];
            c_[nrowsPack + irow] = a_[nrowsPack + irow] - b_[1];
            c_[2*nrowsPack + irow] = a_[2*nrowsPack + irow] - b_[2];
        }
    } else {
        for(uint64_t irow =0; irow<nrowsPack; ++irow){
            c_[irow] = a_[irow] - b_[irow];
            c_[nrowsPack + irow] = a_[nrowsPack + irow] - b_[nrowsPack + irow];
            c_[2*nrowsPack + irow] = a_[2*nrowsPack + irow] - b_[2*nrowsPack + irow];
        }
    }
}

inline void Goldilocks3::mul_pack(uint64_t nrowsPack, Goldilocks::Element *c_, const Goldilocks::Element *a_, const bool const_a, const Goldilocks::Element *b_, const bool const_b)
{
    if(const_a && const_b) {
        for (uint64_t i = 0; i < nrowsPack; ++i)
        {
            Goldilocks::Element A = (a_[0] + a_[1]) * (b_[0] + b_[1]);
            Goldilocks::Element B = (a_[0] + a_[2]) * (b_[0] + b_[2]);
            Goldilocks::Element C = (a_[1] + a_[2]) * (b_[1] + b_[2]);
            Goldilocks::Element D = a_[0] * b_[0];
            Goldilocks::Element E = a_[1] * b_[1];
            Goldilocks::Element F = a_[2] * b_[2];
            Goldilocks::Element G = D - E;

            c_[i] = (C + G) - F;
            c_[nrowsPack + i] = ((((A + C) - E) - E) - D);
            c_[2*nrowsPack + i] = B - G;
        }
    } else if(const_a) {
        for (uint64_t i = 0; i < nrowsPack; ++i)
        {
            Goldilocks::Element A = (a_[0] + a_[1]) * (b_[i] + b_[nrowsPack + i]);
            Goldilocks::Element B = (a_[0] + a_[2]) * (b_[i] + b_[2*nrowsPack + i]);
            Goldilocks::Element C = (a_[1] + a_[2]) * (b_[nrowsPack + i] + b_[2*nrowsPack + i]);
            Goldilocks::Element D = a_[0] * b_[i];
            Goldilocks::Element E = a_[1] * b_[nrowsPack + i];
            Goldilocks::Element F = a_[2] * b_[2*nrowsPack + i];
            Goldilocks::Element G = D - E;

            c_[i] = (C + G) - F;
            c_[nrowsPack + i] = ((((A + C) - E) - E) - D);
            c_[2*nrowsPack + i] = B - G;
        }
    } else if(const_b) {
        for (uint64_t i = 0; i < nrowsPack; ++i)
        {
            Goldilocks::Element A = (a_[i] + a_[nrowsPack + i]) * (b_[0] + b_[1]);
            Goldilocks::Element B = (a_[i] + a_[2*nrowsPack + i]) * (b_[0] + b_[2]);
            Goldilocks::Element C = (a_[nrowsPack + i] + a_[2*nrowsPack + i]) * (b_[1] + b_[2]);
            Goldilocks::Element D = a_[i] * b_[0];
            Goldilocks::Element E = a_[nrowsPack + i] * b_[1];
            Goldilocks::Element F = a_[2*nrowsPack + i] * b_[2];
            Goldilocks::Element G = D - E;

            c_[i] = (C + G) - F;
            c_[nrowsPack + i] = ((((A + C) - E) - E) - D);
            c_[2*nrowsPack + i] = B - G;
        }
    } else {
        for (uint64_t i = 0; i < nrowsPack; ++i)
        {
            Goldilocks::Element A = (a_[i] + a_[nrowsPack + i]) * (b_[i] + b_[nrowsPack + i]);
            Goldilocks::Element B = (a_[i] + a_[2*nrowsPack + i]) * (b_[i] + b_[2*nrowsPack + i]);
            Goldilocks::Element C = (a_[nrowsPack + i] + a_[2*nrowsPack + i]) * (b_[nrowsPack + i] + b_[2*nrowsPack + i]);
            Goldilocks::Element D = a_[i] * b_[i];
            Goldilocks::Element E = a_[nrowsPack + i] * b_[nrowsPack + i];
            Goldilocks::Element F = a_[2*nrowsPack + i] * b_[2*nrowsPack + i];
            Goldilocks::Element G = D - E;

            c_[i] = (C + G) - F;
            c_[nrowsPack + i] = ((((A + C) - E) - E) - D);
            c_[2*nrowsPack + i] = B - G;
        }
    }
};

inline void Goldilocks3::op_pack( uint64_t nrowsPack, uint64_t op, Goldilocks::Element *c, const Goldilocks::Element *a, const bool const_a, const Goldilocks::Element *b, const bool const_b)
{
    switch (op)
    {
    case 0:
        add_pack(nrowsPack, c, a, const_a, b, const_b);
        break;
    case 1:
        sub_pack(nrowsPack, c, a, const_a, b, const_b);
        break;
    case 2:
        mul_pack(nrowsPack, c, a, const_a, b, const_b);
        break;
    case 3:
        sub_pack(nrowsPack, c, b, const_b, a, const_a);
        break;
    default:
        assert(0);
        break;
    }
}

inline void Goldilocks3::op_pack( uint64_t nrowsPack, uint64_t op, Goldilocks::Element *c, const Goldilocks::Element *a, const Goldilocks::Element *b)
{
    switch (op)
    {
    case 0:
        add_pack(nrowsPack, c, a, false, b, false);
        break;
    case 1:
        sub_pack(nrowsPack, c, a, false, b, false);
        break;
    case 2:
        mul_pack(nrowsPack, c, a, false, b, false);
        break;
    case 3:
        sub_pack(nrowsPack, c, b, false, a, false);
        break;
    default:
        assert(0);
        break;
    }
}

inline void Goldilocks3::op_31_pack( uint64_t nrowsPack, uint64_t op, Goldilocks::Element *c, const Goldilocks::Element *a, const Goldilocks::Element *b)
{
    switch (op)
    {
    case 0:
        for (uint64_t i = 0; i < nrowsPack; ++i)
        {
            c[i] = a[i] + b[i];
            c[nrowsPack + i] = a[nrowsPack + i];
            c[2*nrowsPack + i] = a[2*nrowsPack + i];
        }
        break;
    case 1:
        for (uint64_t i = 0; i < nrowsPack; ++i)
        {
            c[i] = a[i] - b[i];
            c[nrowsPack + i] = a[nrowsPack + i];
            c[2*nrowsPack + i] = a[2*nrowsPack + i];
        }
        break;
    case 2:
        for (uint64_t i = 0; i < nrowsPack; ++i)
        {
            c[i] = a[i] * b[i];
            c[nrowsPack + i] = a[nrowsPack + i] * b[i];
            c[2*nrowsPack + i] = a[2*nrowsPack + i] * b[i];
        }
        break;
    case 3:
        for (uint64_t i = 0; i < nrowsPack; ++i)
        {
            c[i] = b[i] - a[i];
            c[nrowsPack + i] = -a[nrowsPack + i];
            c[2*nrowsPack + i] = -a[2*nrowsPack + i];
        }
        break;
    default:
        assert(0);
        break;
    }
}

inline void Goldilocks3::op_31_pack(uint64_t nrowsPack, uint64_t op, Goldilocks::Element *c, const Goldilocks::Element *a, const bool const_a, const Goldilocks::Element *b, const bool const_b)
{
    if (const_a && const_b) {
        switch (op)
        {
        case 0:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[0] + b[0];
                c[nrowsPack + i] = a[1];
                c[2*nrowsPack + i] = a[2];
            }
            break;
        case 1:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[0] - b[0];
                c[nrowsPack + i] = a[1];
                c[2*nrowsPack + i] = a[2];
            }
            break;
        case 2:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[0] * b[0];
                c[nrowsPack + i] = a[1] * b[0];
                c[2*nrowsPack + i] = a[2] * b[0];
            }
            break;
        case 3:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = b[0] - a[0];
                c[nrowsPack + i] = -a[1];
                c[2*nrowsPack + i] = -a[2];
            }
            break;
        default:
            assert(0);
            break;
        }
    } else if(const_a) {
        switch (op)
        {
        case 0:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[0] + b[i];
                c[nrowsPack + i] = a[1];
                c[2*nrowsPack + i] = a[2];
            }
            break;
        case 1:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[0] - b[i];
                c[nrowsPack + i] = a[1];
                c[2*nrowsPack + i] = a[2];
            }
            break;
        case 2:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[0] * b[i];
                c[nrowsPack + i] = a[1] * b[i];
                c[2*nrowsPack + i] = a[2] * b[i];
            }
            break;
        case 3:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = b[i] - a[0];
                c[nrowsPack + i] = -a[1];
                c[2*nrowsPack + i] = -a[2];
            }
            break;
        default:
            assert(0);
            break;
        }
    } else if(const_b) {
        switch (op)
        {
        case 0:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[i] + b[0];
                c[nrowsPack + i] = a[nrowsPack + i];
                c[2*nrowsPack + i] = a[2*nrowsPack + i];
            }
            break;
        case 1:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[i] - b[0];
                c[nrowsPack + i] = a[nrowsPack + i];
                c[2*nrowsPack + i] = a[2*nrowsPack + i];
            }
            break;
        case 2:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[i] * b[0];
                c[nrowsPack + i] = a[nrowsPack + i] * b[0];
                c[2*nrowsPack + i] = a[2*nrowsPack + i] * b[0];
            }
            break;
        case 3:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = b[0] - a[i];
                c[nrowsPack + i] = -a[nrowsPack + i];
                c[2*nrowsPack + i] = -a[2*nrowsPack + i];
            }
            break;
        default:
            assert(0);
            break;
        }
    } else {
        switch (op)
        {
        case 0:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[i] + b[i];
                c[nrowsPack + i] = a[nrowsPack + i];
                c[2*nrowsPack + i] = a[2*nrowsPack + i];
            }
            break;
        case 1:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[i] - b[i];
                c[nrowsPack + i] = a[nrowsPack + i];
                c[2*nrowsPack + i] = a[2*nrowsPack + i];
            }
            break;
        case 2:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[i] * b[i];
                c[nrowsPack + i] = a[nrowsPack + i] * b[i];
                c[2*nrowsPack + i] = a[2*nrowsPack + i] * b[i];
            }
            break;
        case 3:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = b[i] - a[i];
                c[nrowsPack + i] = -a[nrowsPack + i];
                c[2*nrowsPack + i] = -a[2*nrowsPack + i];
            }
            break;
        default:
            assert(0);
            break;
        }
    }
    
}

#endif