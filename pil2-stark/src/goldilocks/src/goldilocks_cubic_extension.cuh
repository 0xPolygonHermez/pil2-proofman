#ifndef GOLDILOCKS_F3_CUH
#define GOLDILOCKS_F3_CUH

#include <stdint.h> // for uint64_t
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include <cassert>
#include <vector>
#include "gl64_t.cuh"
#include "cuda_utils.cuh"

#ifdef __USE_CUDA__
#ifdef __GNUC__
#define asm __asm__ __volatile__
#else
#define asm asm volatile
#endif
#endif

#define FIELD_EXTENSION 3

/*
This is a field extension 3 of the goldilocks:
Prime: 0xFFFFFFFF00000001
Irreducible polynomial: x^3 - x -1
*/

__device__ __forceinline__ gl64_t neg_element(gl64_t x)
{
    gl64_t z = 0ul;
    return z - x;
}

// TODO: Review and optimize inv imlementation
__device__ __forceinline__ gl64_t inv_element(const gl64_t &in1)
{
    if (in1.is_zero())
    {
        assert(0);
    }
    u_int64_t t = 0;
    u_int64_t r = GOLDILOCKS_PRIME;
    u_int64_t newt = 1;

    u_int64_t newr = (in1 >= GOLDILOCKS_PRIME) ? in1.get_val() - GOLDILOCKS_PRIME : in1.get_val();
    gl64_t q;
    gl64_t aux1;
    gl64_t aux2;
    while (newr != 0)
    {
        q = r / newr;
        aux1 = t;
        aux2 = newt;
        t = aux2;
        newt = aux1 - q * aux2;
        aux1 = r;
        aux2 = newr;
        r = aux2;
        newr = aux1 - q * aux2;
    }

    return t;
}

class Goldilocks3GPU
{
public:
    typedef gl64_t Element[FIELD_EXTENSION];

private:
    static const Element ZERO;
    static const Element ONE;
    static const Element NEGONE;

public:
    uint64_t m = 1 * FIELD_EXTENSION;
    uint64_t p = GOLDILOCKS_PRIME;
    uint64_t n64 = 1;
    uint64_t n32 = n64 * 2;
    uint64_t n8 = n32 * 4;

    static const Element &zero() { return ZERO; };

    static __device__ __forceinline__ void zero(Element &result)
    {
        result[0] = 0ul;
        result[1] = 0ul;
        result[2] = 0ul;
    };

    static __device__ __forceinline__ const Element &one() { return ONE; };

    static __device__ __forceinline__ void one(Element &result)
    {
        result[0] = 1ul;
        result[1] = 0ul;
        result[2] = 0ul;
    };

    static __device__ __forceinline__ bool isOne(Element &result)
    {
        return result[0].is_one() && result[1].is_zero() && result[2].is_zero();
    };

    static void copy(Element &dst, const Element &src)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            dst[i] = src[i];
        }
    };
    static void copy(Element *dst, const Element *src)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            (*dst)[i] = (*src)[i];
        }
    };

    static __device__ __forceinline__ void fromU64(Element &result, uint64_t in1[FIELD_EXTENSION])
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = in1[i];
        }
    }
    static __device__ __forceinline__ void fromS32(Element &result, int32_t in1[FIELD_EXTENSION])
    {
        //  (in1 < 0) ? aux = static_cast<uint64_t>(in1) + GOLDILOCKS_PRIME : aux = static_cast<uint64_t>(in1);
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = (in1[i] < 0) ? static_cast<uint64_t>(in1[i]) + GOLDILOCKS_PRIME : static_cast<uint64_t>(in1[i]);
        }
    }
    static __device__ __forceinline__ void toU64(uint64_t (&result)[FIELD_EXTENSION], const Element &in1)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = (in1[i] >= GOLDILOCKS_PRIME) ? in1[i].get_val() - GOLDILOCKS_PRIME : in1[i].get_val();
        }
    }

    // ======== ADD ========
    static __device__ __forceinline__ void add(Element &result, const Element &a, const uint64_t &b)
    {
        result[0] = a[0] + gl64_t(b);
        result[1] = a[1];
        result[2] = a[2];
    }
    static __device__ __forceinline__ void add(Element &result, const Element &a, const gl64_t b)
    {
        result[0] = a[0] + b;
        result[1] = a[1];
        result[2] = a[2];
    }
    static __device__ __forceinline__ void add(Element &result, const gl64_t a, const Element &b)
    {
        add(result, b, a);
    }
    static __device__ __forceinline__ void add(Element &result, const Element &a, const Element &b)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = a[i] + b[i];
        }
    }

    // ======== SUB ========
    static __device__ __forceinline__ void sub(Element &result, Element &a, uint64_t &b)
    {
        result[0] = a[0] - gl64_t(b);
        result[1] = a[1];
        result[2] = a[2];
    }
    static __device__ __forceinline__ void sub(Element &result, gl64_t a, Element const &b)
    {
        result[0] = a - b[0];
        result[1] = neg_element(b[1]);
        result[2] = neg_element(b[2]);
    }
    static __device__ __forceinline__ void sub(Element &result, Element &a, gl64_t b)
    {
        result[0] = a[0] - b;
        result[1] = a[1];
        result[2] = a[2];
    }
    static __device__ __forceinline__ void sub(Element &result, Element &a, Element &b)
    {
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++)
        {
            result[i] = a[i] - b[i];
        }
    }

    // ======== NEG ========
    static __device__ __forceinline__ void neg(Element &result, Element &a)
    {
        sub(result, (Element &)zero(), a);
    }

    // ======== MUL ========
    static __device__ __forceinline__ void mul(Element *result, Element *a, Element *b)
    {
        mul(*result, *a, *b);
    }
    static __device__ __forceinline__ void mul(Element &result, Element &a, Element &b)
    {
        gl64_t A = (a[0] + a[1]) * (b[0] + b[1]);
        gl64_t B = (a[0] + a[2]) * (b[0] + b[2]);
        gl64_t C = (a[1] + a[2]) * (b[1] + b[2]);
        gl64_t D = a[0] * b[0];
        gl64_t E = a[1] * b[1];
        gl64_t F = a[2] * b[2];
        gl64_t G = D - E;

        result[0] = (C + G) - F;
        result[1] = ((((A + C) - E) - E) - D);
        result[2] = B - G;
    };
    static __device__ __forceinline__ void mul(Element &result, Element &a, gl64_t &b)
    {
        result[0] = a[0] * b;
        result[1] = a[1] * b;
        result[2] = a[2] * b;
    }
    static __device__ __forceinline__ void mul(Element &result, gl64_t a, Element &b)
    {
        mul(result, b, a);
    }
    static __device__ __forceinline__ void mul(Element &result, Element &a, uint64_t b)
    {
        result[0] = a[0] * b;
        result[1] = a[1] * b;
        result[2] = a[2] * b;
    }

    // ======== DIV ========
    static __device__ __forceinline__ void div(Element &result, Element &a, gl64_t b)
    {
        gl64_t b_inv = inv_element(b);
        mul(result, a, b_inv);
    }

    // ======== MULSCALAR ========
    // TBD

    // ======== SQUARE ========
    static __device__ __forceinline__ void square(Element &result, Element &a)
    {
        mul(result, a, a);
    }

    // ======== INV ========
    static __device__ __forceinline__ void inv(Element *result, Element *a)
    {
        inv(*result, *a);
    }
    static __device__ __noinline__ void inv(Element &result, Element &a)
    {
        gl64_t aa = a[0] * a[0];
        gl64_t ac = a[0] * a[2];
        gl64_t ba = a[1] * a[0];
        gl64_t bb = a[1] * a[1];
        gl64_t bc = a[1] * a[2];
        gl64_t cc = a[2] * a[2];

        gl64_t aaa = aa * a[0];
        gl64_t aac = aa * a[2];
        gl64_t abc = ba * a[2];
        gl64_t abb = ba * a[1];
        gl64_t acc = ac * a[2];
        gl64_t bbb = bb * a[1];
        gl64_t bcc = bc * a[2];
        gl64_t ccc = cc * a[2];

        gl64_t t = abc + abc + abc + abb - aaa - aac - aac - acc - bbb + bcc - ccc;

        gl64_t tinv = inv_element(t);
        gl64_t i1 = (bc + bb - aa - ac - ac - cc) * tinv;

        gl64_t i2 = (ba - cc) * tinv;
        gl64_t i3 = (ac + cc - bb) * tinv;

        result[0] = i1;
        result[1] = i2;
        result[2] = i3;
    }

    // ======== POW ========
    static __device__ __forceinline__ void pow(Element &base, uint64_t exp, Element &result)
    {
        one(result);
        while (exp > 0)
        {
            if (exp % 2 == 1)
            {
                mul(result, result, base);
            }
            mul(base, base, base);
            exp /= 2;
        }
    }

    // ======== EXPRESSIONS ========
    static __device__ __forceinline__ void copy_gpu(gl64_t *c_, const gl64_t *a_, bool const_a)
    {
        if (const_a)
        {
            c_[threadIdx.x] = a_[0];
            c_[blockDim.x + threadIdx.x] = a_[1];
            c_[2 * blockDim.x + threadIdx.x] = a_[2];
        }
        else
        {
            c_[threadIdx.x] = a_[threadIdx.x];
            c_[blockDim.x + threadIdx.x] = a_[blockDim.x + threadIdx.x];
            c_[2 * blockDim.x + threadIdx.x] = a_[2 * blockDim.x + threadIdx.x];
        }
    }
    static __device__ __forceinline__ void add_gpu(gl64_t *c_, const gl64_t *a_, bool const_a, const gl64_t *b_, bool const_b)
    {

        if (const_a && const_b)
        {
            c_[threadIdx.x] = a_[0] + b_[0];
            c_[blockDim.x + threadIdx.x] = a_[1] + b_[1];
            c_[2 * blockDim.x + threadIdx.x] = a_[2] + b_[2];
        }
        else if (const_a)
        {
            c_[threadIdx.x] = a_[0] + b_[threadIdx.x];
            c_[blockDim.x + threadIdx.x] = a_[1] + b_[blockDim.x + threadIdx.x];
            c_[2 * blockDim.x + threadIdx.x] = a_[2] + b_[2 * blockDim.x + threadIdx.x];
        }
        else if (const_b)
        {
            c_[threadIdx.x] = a_[threadIdx.x] + b_[0];
            c_[blockDim.x + threadIdx.x] = a_[blockDim.x + threadIdx.x] + b_[1];
            c_[2 * blockDim.x + threadIdx.x] = a_[2 * blockDim.x + threadIdx.x] + b_[2];
        }
        else
        {
            c_[threadIdx.x] = a_[threadIdx.x] + b_[threadIdx.x];
            c_[blockDim.x + threadIdx.x] = a_[blockDim.x + threadIdx.x] + b_[blockDim.x + threadIdx.x];
            c_[2 * blockDim.x + threadIdx.x] = a_[2 * blockDim.x + threadIdx.x] + b_[2 * blockDim.x + threadIdx.x];
        }
    }
    static __device__ __forceinline__ void sub_gpu(gl64_t *c_, const gl64_t *a_, bool const_a, const gl64_t *b_, bool const_b)
    {
        if (const_a && const_b)
        {
            c_[threadIdx.x] = a_[0] - b_[0];
            c_[blockDim.x + threadIdx.x] = a_[1] - b_[1];
            c_[2 * blockDim.x + threadIdx.x] = a_[2] - b_[2];
        }
        else if (const_a)
        {
            c_[threadIdx.x] = a_[0] - b_[threadIdx.x];
            c_[blockDim.x + threadIdx.x] = a_[1] - b_[blockDim.x + threadIdx.x];
            c_[2 * blockDim.x + threadIdx.x] = a_[2] - b_[2 * blockDim.x + threadIdx.x];
        }
        else if (const_b)
        {
            c_[threadIdx.x] = a_[threadIdx.x] - b_[0];
            c_[blockDim.x + threadIdx.x] = a_[blockDim.x + threadIdx.x] - b_[1];
            c_[2 * blockDim.x + threadIdx.x] = a_[2 * blockDim.x + threadIdx.x] - b_[2];
        }
        else
        {
            c_[threadIdx.x] = a_[threadIdx.x] - b_[threadIdx.x];
            c_[blockDim.x + threadIdx.x] = a_[blockDim.x + threadIdx.x] - b_[blockDim.x + threadIdx.x];
            c_[2 * blockDim.x + threadIdx.x] = a_[2 * blockDim.x + threadIdx.x] - b_[2 * blockDim.x + threadIdx.x];
        }
    }
    static __device__ __forceinline__ void mul_gpu(gl64_t *c_, const gl64_t *a_, bool const_a, const gl64_t *b_, bool const_b)
    {

        gl64_t A, B, C, D, E, F, G;

        if (const_a && const_b)
        {
            A = (a_[0] + a_[1]) * (b_[0] + b_[1]);
            B = (a_[0] + a_[2]) * (b_[0] + b_[2]);
            C = (a_[1] + a_[2]) * (b_[1] + b_[2]);
            D = a_[0] * b_[0];
            E = a_[1] * b_[1];
            F = a_[2] * b_[2];
        }
        else if (const_a)
        {
            A = (a_[0] + a_[1]) * (b_[threadIdx.x] + b_[blockDim.x + threadIdx.x]);
            B = (a_[0] + a_[2]) * (b_[threadIdx.x] + b_[2 * blockDim.x + threadIdx.x]);
            C = (a_[1] + a_[2]) * (b_[blockDim.x + threadIdx.x] + b_[2 * blockDim.x + threadIdx.x]);
            D = a_[0] * b_[threadIdx.x];
            E = a_[1] * b_[blockDim.x + threadIdx.x];
            F = a_[2] * b_[2 * blockDim.x + threadIdx.x];
        }
        else if (const_b)
        {
            A = (a_[threadIdx.x] + a_[blockDim.x + threadIdx.x]) * (b_[0] + b_[1]);
            B = (a_[threadIdx.x] + a_[2 * blockDim.x + threadIdx.x]) * (b_[0] + b_[2]);
            C = (a_[blockDim.x + threadIdx.x] + a_[2 * blockDim.x + threadIdx.x]) * (b_[1] + b_[2]);
            D = a_[threadIdx.x] * b_[0];
            E = a_[blockDim.x + threadIdx.x] * b_[1];
            F = a_[2 * blockDim.x + threadIdx.x] * b_[2];
        }
        else
        {
            A = (a_[threadIdx.x] + a_[blockDim.x + threadIdx.x]) * (b_[threadIdx.x] + b_[blockDim.x + threadIdx.x]);
            B = (a_[threadIdx.x] + a_[2 * blockDim.x + threadIdx.x]) * (b_[threadIdx.x] + b_[2 * blockDim.x + threadIdx.x]);
            C = (a_[blockDim.x + threadIdx.x] + a_[2 * blockDim.x + threadIdx.x]) * (b_[blockDim.x + threadIdx.x] + b_[2 * blockDim.x + threadIdx.x]);
            D = a_[threadIdx.x] * b_[threadIdx.x];
            E = a_[blockDim.x + threadIdx.x] * b_[blockDim.x + threadIdx.x];
            F = a_[2 * blockDim.x + threadIdx.x] * b_[2 * blockDim.x + threadIdx.x];
        }

        G = D - E;
        c_[threadIdx.x] = (C + G) - F;
        c_[blockDim.x + threadIdx.x] = ((((A + C) - E) - E) - D);
        c_[2 * blockDim.x + threadIdx.x] = B - G;
    }

    static __device__ __forceinline__ void op_gpu(uint64_t op, gl64_t *c, const gl64_t *a, bool const_a, const gl64_t *b, bool const_b)
    {
        switch (op)
        {
        case 0:
            add_gpu(c, a, const_a, b, const_b);
            break;
        case 1:
            sub_gpu(c, a, const_a, b, const_b);
            break;
        case 2:
            mul_gpu(c, a, const_a, b, const_b);
            break;
        case 3:
            sub_gpu(c, b, const_b, a, const_a);
            break;
        default:
            assert(0);
            break;
        }
    }
    static __device__ __forceinline__ void op_31_gpu(uint64_t op, gl64_t *c, const gl64_t *a, bool const_a, const gl64_t *b, bool const_b)
    {

        if (const_a && const_b)
        {
            switch (op)
            {
            case 0:
                c[threadIdx.x] = a[0] + b[0];
                c[blockDim.x + threadIdx.x] = a[1];
                c[2 * blockDim.x + threadIdx.x] = a[2];
                break;
            case 1:
                c[threadIdx.x] = a[0] - b[0];
                c[blockDim.x + threadIdx.x] = a[1];
                c[2 * blockDim.x + threadIdx.x] = a[2];
                break;
            case 2:
                c[threadIdx.x] = a[0] * b[0];
                c[blockDim.x + threadIdx.x] = a[1] * b[0];
                c[2 * blockDim.x + threadIdx.x] = a[2] * b[0];
                break;
            case 3:
                c[threadIdx.x] = b[0] - a[0];
                c[blockDim.x + threadIdx.x] = -a[1];
                c[2 * blockDim.x + threadIdx.x] = -a[2];
                break;
            default:
                assert(0);
                break;
            }
        }
        else if (const_a)
        {
            switch (op)
            {
            case 0:
                c[threadIdx.x] = a[0] + b[threadIdx.x];
                c[blockDim.x + threadIdx.x] = a[1];
                c[2 * blockDim.x + threadIdx.x] = a[2];
                break;
            case 1:
                c[threadIdx.x] = a[0] - b[threadIdx.x];
                c[blockDim.x + threadIdx.x] = a[1];
                c[2 * blockDim.x + threadIdx.x] = a[2];
                break;
            case 2:
                c[threadIdx.x] = a[0] * b[threadIdx.x];
                c[blockDim.x + threadIdx.x] = a[1] * b[threadIdx.x];
                c[2 * blockDim.x + threadIdx.x] = a[2] * b[threadIdx.x];
                break;
            case 3:
                c[threadIdx.x] = b[threadIdx.x] - a[0];
                c[blockDim.x + threadIdx.x] = -a[1];
                c[2 * blockDim.x + threadIdx.x] = -a[2];
                break;
            default:
                assert(0);
                break;
            }
        }
        else if (const_b)
        {
            switch (op)
            {
            case 0:
                c[threadIdx.x] = a[threadIdx.x] + b[0];
                c[blockDim.x + threadIdx.x] = a[blockDim.x + threadIdx.x];
                c[2 * blockDim.x + threadIdx.x] = a[2 * blockDim.x + threadIdx.x];
                break;
            case 1:
                c[threadIdx.x] = a[threadIdx.x] - b[0];
                c[blockDim.x + threadIdx.x] = a[blockDim.x + threadIdx.x];
                c[2 * blockDim.x + threadIdx.x] = a[2 * blockDim.x + threadIdx.x];
                break;
            case 2:
                c[threadIdx.x] = a[threadIdx.x] * b[0];
                c[blockDim.x + threadIdx.x] = a[blockDim.x + threadIdx.x] * b[0];
                c[2 * blockDim.x + threadIdx.x] = a[2 * blockDim.x + threadIdx.x] * b[0];
                break;
            case 3:
                c[threadIdx.x] = b[0] - a[threadIdx.x];
                c[blockDim.x + threadIdx.x] = -a[blockDim.x + threadIdx.x];
                c[2 * blockDim.x + threadIdx.x] = -a[2 * blockDim.x + threadIdx.x];
                break;
            default:
                assert(0);
                break;
            }
        }
        else
        {
            switch (op)
            {
            case 0:
                c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
                c[blockDim.x + threadIdx.x] = a[blockDim.x + threadIdx.x];
                c[2 * blockDim.x + threadIdx.x] = a[2 * blockDim.x + threadIdx.x];
                break;
            case 1:
                c[threadIdx.x] = a[threadIdx.x] - b[threadIdx.x];
                c[blockDim.x + threadIdx.x] = a[blockDim.x + threadIdx.x];
                c[2 * blockDim.x + threadIdx.x] = a[2 * blockDim.x + threadIdx.x];
                break;
            case 2:
                c[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
                c[blockDim.x + threadIdx.x] = a[blockDim.x + threadIdx.x] * b[threadIdx.x];
                c[2 * blockDim.x + threadIdx.x] = a[2 * blockDim.x + threadIdx.x] * b[threadIdx.x];
                break;
            case 3:
                c[threadIdx.x] = b[threadIdx.x] - a[threadIdx.x];
                c[blockDim.x + threadIdx.x] = -a[blockDim.x + threadIdx.x];
                c[2 * blockDim.x + threadIdx.x] = -a[2 * blockDim.x + threadIdx.x];
                break;
            default:
                assert(0);
                break;
            }
        }
    }
};

#endif // GOLDILOCKS_F3_CUH
