// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_FF_GL64_T_CUH__
#define __SPPARK_FF_GL64_T_CUH__

#include <cstdint>
#include <cassert>

namespace gl64_device
{
    static __device__ __constant__ /*const*/ uint32_t W = 0xffffffffU;
}

#ifdef __USE_CUDA__
#define inline __device__ __forceinline__
#ifdef __GNUC__
#define asm __asm__ __volatile__
#else
#define asm asm volatile
#endif

#ifdef GL64_PARTIALLY_REDUCED
//
// This variant operates with partially reduced values, ones less than
// 2**64, as opposed to less than 2**64-2**32+1. For this reason the
// final results need to be fixed up.
//
// Expected gain is that the fixups in the beginnings of additions and
// subtractions can be interleaved with preceding multiplication, hence
// folding multiple critical paths. This is provided that multiplication
// result is passed as 2nd[!] argument to addition/subtraction.
//
// It should be noted that either multiplication variant can handle
// partially reduced inputs. This is used in exponentiation.
//
#endif

class gl64_t
{
public:
    uint64_t val;

public:
    static const uint32_t degree = 1;
    static const unsigned nbits = 64;
    static const uint64_t MOD = 0xffffffff00000001U;
    static constexpr size_t __device__ bit_length() { return 64; }

    inline uint64_t &operator[](size_t i) { return val; }
    inline const uint64_t &operator[](size_t i) const { return val; }
    inline size_t len() const { return 1; }

    inline gl64_t() {}
    inline gl64_t(const uint64_t a)
    {
        val = a;
        to();
    }
    inline gl64_t(const uint64_t *p)
    {
        val = *p;
        to();
    }

    inline uint64_t get_val() const { return val; }

    inline void set_val(uint64_t x) { val = x; }

    inline operator uint64_t() const
    {
        auto ret = *this;
        ret.from();
        return ret.val;
    }
    inline void store(uint64_t *p) const
    {
        *p = *this;
    }

    inline gl64_t &operator&=(const uint64_t &b)
    {
        this->val &= b;
        return *this;
    }

    inline gl64_t &operator+=(const gl64_t &b)
    {
        from();

        uint64_t tmp;
        uint32_t carry;

        asm("add.cc.u64 %0, %0, %2; addc.u32 %1, 0, 0;"
            : "+l"(val), "=r"(carry)
            : "l"(b.val));

        asm("{ .reg.pred %top;");
# ifdef GL64_PARTIALLY_REDUCED
        asm("sub.u64 %0, %1, %2;"
            : "=l"(tmp)
            : "l"(val), "l"(MOD));
        asm("setp.ne.u32 %top, %0, 0;" :: "r"(carry));
        asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
# else
        asm("sub.cc.u64 %0, %2, %3; subc.u32 %1, %1, 0;"
            : "=l"(tmp), "+r"(carry)
            : "l"(val), "l"(MOD));
        asm("setp.eq.u32 %top, %0, 0;" :: "r"(carry));
        asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
        asm("setp.ge.u64 %top, %0, %1;" : : "l"(val), "l"(MOD));
        asm("@%top sub.u64 %0, %0, %1;" : "+l"(val) : "l"(MOD));
# endif
        asm("}");        
        return *this;
    }
    friend inline gl64_t operator+(gl64_t a, const gl64_t &b)
    {
        return a += b;
    }

    inline gl64_t &operator<<=(unsigned l)
    {
        from();

        uint64_t tmp;
        uint32_t carry;
        asm("{ .reg.pred %top;");

        while (l--)
        {
            asm("add.cc.u64 %0, %0, %0; addc.u32 %1, 0, 0;"
                : "+l"(val), "=r"(carry));
            asm("sub.cc.u64 %0, %2, %3; subc.u32 %1, %1, 0;"
                : "=l"(tmp), "+r"(carry)
                : "l"(val), "l"(MOD));
            asm("setp.eq.u32 %top, %0, 0;" ::"r"(carry));
            asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
        }

        asm("}");
        return *this;
    }
    friend inline gl64_t operator<<(gl64_t a, unsigned l)
    {
        return a <<= l;
    }

    inline gl64_t &operator>>=(unsigned r)
    {
        uint64_t tmp;
        uint32_t carry;

        while (r--)
        {
            tmp = val & 1 ? MOD : 0;
            asm("add.cc.u64 %0, %0, %2; addc.u32 %1, 0, 0;"
                : "+l"(tmp) "=r"(carry)
                : "l"(val));
            val = (tmp >> 1) + ((uint64_t)carry << 63);
        }

        return *this;
    }
    friend inline gl64_t operator>>(gl64_t a, unsigned r)
    {
        return a >>= r;
    }

    inline gl64_t &operator-=(const gl64_t &b)
    {
        uint64_t tmp;
        uint32_t borrow;
        asm("{ .reg.pred %top;");

#ifdef GL64_PARTIALLY_REDUCED
        asm("add.cc.u64 %0, %2, %3; addc.u32 %1, 0, 0;"
            : "=l"(tmp), "=r"(borrow)
            : "l"(val), "l"(MOD));
        asm("setp.eq.u32 %top, %0, 0;" ::"r"(borrow));
        asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
#endif

        asm("mov.b64 %0, %1;" 
            : "=l"(tmp) 
            : "l"(b.val));

        asm("setp.ge.u64 %top, %1, %2;"  // Set predicate if tmp >= MOD
            "@%top sub.u64 %0, %1, %2;"   // If true, subtract MOD from tmp
            : "+l"(tmp)
            : "l"(tmp), "l"(MOD));

        asm("sub.cc.u64 %0, %0, %2; subc.u32 %1, 0, 0;"
            : "+l"(val), "=r"(borrow)
            : "l"(tmp));
        asm("add.u64 %0, %1, %2;" : "=l"(tmp) : "l"(val), "l"(MOD));
        asm("setp.ne.u32 %top, %0, 0;" ::"r"(borrow));
        asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
        asm("}");
        return *this;
    }
    friend inline gl64_t operator-(gl64_t a, const gl64_t &b)
    {
        return a -= b;
    }

    inline gl64_t &cneg(bool flag)
    {
        uint64_t tmp;

#ifdef GL64_PARTIALLY_REDUCED
        uint32_t borrow;

        asm("sub.cc.u64 %0, %2, %3; subc.u32 %1, 0, 0;"
            : "=l"(tmp), "=r"(borrow)
            : "l"(MOD), "l"(val));

        asm("{ .reg.pred %flag;");
        asm("setp.ne.u32 %flag, %0, 0;" ::"r"(borrow));
        asm("@%flag add.u64 %0, %0, %1;" : "+l"(tmp) : "l"(MOD));
        asm("setp.ne.u32 %flag, %0, 0;" ::"r"((int)flag));
        asm("@%flag mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
        asm("}");
#else
        int is_zero = (val == 0);

        asm("sub.u64 %0, %1, %2;" : "=l"(tmp) : "l"(MOD), "l"(val));
        asm("{ .reg.pred %flag;");
        asm("setp.ne.u32 %flag, %0, 0;" ::"r"((int)flag));
        asm("@%flag setp.eq.u32 %flag, %0, 0;" ::"r"(is_zero));
        asm("@%flag mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
        asm("}");
#endif

        return *this;
    }
    friend inline gl64_t cneg(gl64_t a, bool flag)
    {
        return a.cneg(flag);
    }
    inline gl64_t operator-() const
    {
        gl64_t ret = *this;
        return ret.cneg(true);
    }

    static inline const gl64_t one()
    {
        gl64_t ret;
        ret.val = 1;
        return ret;
    }

#ifdef GL64_PARTIALLY_REDUCED
    inline bool is_zero() const { return val == 0 | val == MOD; }
    inline bool is_one() const { return val == 1 | val == MOD + 1; }
#else
    inline bool is_zero() const { return val == 0; }
    inline bool is_one() const { return val == 1; }
#endif

    inline void zero()
    {
        val = 0;
    }

    friend inline gl64_t czero(const gl64_t &a, int set_z)
    {
        gl64_t ret;
        asm("{ .reg.pred %set_z;");
        asm("setp.ne.s32 %set_z, %0, 0;" : : "r"(set_z));
        asm("selp.u64 %0, 0, %1, %set_z;" : "=l"(ret.val) : "l"(a.val));
        asm("}");
        return ret;
    }

    static inline gl64_t csel(const gl64_t &a, const gl64_t &b, int sel_a)
    {
        gl64_t ret;
        asm("{ .reg.pred %sel_a;");
        asm("setp.ne.s32 %sel_a, %0, 0;" : : "r"(sel_a));
        asm("selp.u64 %0, %1, %2, %sel_a;" : "=l"(ret.val) : "l"(a.val), "l"(b.val));
        asm("}");
        return ret;
    }

private:
    inline uint32_t lo() const { return (uint32_t)(val); }
    inline uint32_t hi() const { return (uint32_t)(val >> 32); }

    inline void mul(const gl64_t &b)
    {
        uint32_t a0 = lo(), b0 = b.lo();
        uint32_t a1 = hi(), b1 = b.hi();
        uint32_t temp[4];

        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(temp[0]), "=r"(temp[1])
            : "r"(a0), "r"(b0));
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(temp[2]), "=r"(temp[3])
            : "r"(a1), "r"(b1));
#if 1
        uint32_t carry; // isolate the first carry in hope compiler uses 3-way iadd3.x
        asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, 0, 0;"
            : "+r"(temp[1]), "+r"(temp[2]), "=r"(carry)
            : "r"(a0), "r"(b1));
        asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, %2, %5;"
            : "+r"(temp[1]), "+r"(temp[2]), "+r"(temp[3])
            : "r"(a1), "r"(b0), "r"(carry));
#else
        asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, %2, 0;"
            : "+r"(temp[1]), "+r"(temp[2]), "+r"(temp[3])
            : "r"(a0), "r"(b1));
        asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, %2, 0;"
            : "+r"(temp[1]), "+r"(temp[2]), "+r"(temp[3])
            : "r"(a1), "r"(b0));
#endif

        reduce(temp);
    }

    inline void reduce(uint32_t temp[4])
    {
        uint32_t carry;
#if __CUDA_ARCH__ >= 700
        asm("sub.cc.u32 %0, %0, %3; subc.cc.u32 %1, %1, %4; subc.u32 %2, 0, 0;"
            : "+r"(temp[0]), "+r"(temp[1]), "=r"(carry)
            : "r"(temp[2]), "r"(temp[3]));
        asm("add.cc.u32 %0, %0, %2; addc.u32 %1, %1, %3;"
            : "+r"(temp[1]), "+r"(carry)
            : "r"(temp[2]), "r"(temp[3]));

        asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, 0, 0;"
            : "+r"(temp[0]), "+r"(temp[1]), "=r"(temp[2])
            : "r"(carry), "r"(gl64_device::W));
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
            : "+r"(temp[0]), "+r"(temp[1])
            : "r"(temp[2]), "r"(gl64_device::W));
#else
        uint32_t b0, b1;
        asm("add.cc.u32 %0, %2, %3; addc.u32 %1, 0, 0;"
            : "=r"(b0), "=r"(b1)
            : "r"(temp[2]), "r"(temp[3]));
        asm("sub.cc.u32 %0, %0, %3; subc.cc.u32 %1, %1, %4; subc.u32 %2, 0, 0;"
            : "+r"(temp[0]), "+r"(temp[1]), "=r"(carry)
            : "r"(b0), "r"(b1));
        asm("add.cc.u32 %0, %0, %2; addc.u32 %1, %1, %3;"
            : "+r"(temp[0]), "+r"(temp[1])
            : "r"(-carry), "r"(carry));
        asm("add.cc.u32 %0, %0, %1; addc.u32 %1, 0, 0;"
            : "+r"(temp[1]), "+r"(temp[2]));

#if __CUDA_ARCH__ >= 700
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
            : "+r"(temp[0]), "+r"(temp[1])
            : "r"(temp[2]), "r"(gl64_device::W));
#else
        asm("add.cc.u32 %0, %0, %2; addc.u32 %1, %1, 0;"
            : "+r"(temp[0]), "+r"(temp[1])
            : "r"(-temp[2]));
#endif
#endif
        asm("mov.b64 %0, {%1, %2};" : "=l"(val) : "r"(temp[0]), "r"(temp[1]));
    }

public:
    friend inline gl64_t operator*(gl64_t a, const gl64_t &b)
    {
        a.mul(b);
        a.to();
        return a;
    }
#ifndef GL64_NO_REDUCTION_KLUDGE
    inline gl64_t &operator*=(const gl64_t &a)
    {
        mul(a);
        to();
        return *this;
    }
#else
    inline gl64_t &operator*=(const gl64_t &a)
    {
        mul(a);
        return *this;
    }
#endif

    // raise to a variable power, variable in respect to threadIdx,
    // but mind the ^ operator's precedence!
    inline gl64_t &operator^=(uint32_t p)
    {
        gl64_t sqr = *this;
        *this = csel(*this, one(), p & 1);

#pragma unroll 1
        while (p >>= 1)
        {
            sqr.mul(sqr);
            if (p & 1)
                mul(sqr);
        }
        to();

        return *this;
    }
    friend inline gl64_t operator^(gl64_t a, uint32_t p)
    {
        return a ^= p;
    }
    inline gl64_t operator()(uint32_t p)
    {
        return *this ^ p;
    }

    // raise to a constant power, e.g. x^7, to be unrolled at compile time
    inline gl64_t &operator^=(int p)
    {
        if (p < 2)
            asm("trap;");

        gl64_t sqr = *this;
        if ((p & 1) == 0)
        {
            do
            {
                sqr.mul(sqr);
                p >>= 1;
            } while ((p & 1) == 0);
            *this = sqr;
        }
        for (p >>= 1; p; p >>= 1)
        {
            sqr.mul(sqr);
            if (p & 1)
                mul(sqr);
        }
        to();

        return *this;
    }
    friend inline gl64_t operator^(gl64_t a, int p)
    {
        return a ^= p;
    }
    inline gl64_t operator()(int p)
    {
        return *this ^ p;
    }
    friend inline gl64_t sqr(gl64_t a)
    {
        return a.sqr();
    }
    inline gl64_t &sqr()
    {
        mul(*this);
        to();
        return *this;
    }

private:
    inline void mul(uint32_t b)
    {
        uint32_t a0 = lo();
        uint32_t a1 = hi();
        uint32_t temp[3];

        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(temp[0]), "=r"(temp[1])
            : "r"(a0), "r"(b));

        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, 0;"
            : "+r"(temp[1]), "=r"(temp[2])
            : "r"(a1), "r"(b));

        asm("sub.cc.u32 %0, 0, %2; subc.u32 %1, %2, 0;"
            : "=r"(a0), "=r"(a1)
            : "r"(temp[2]));
        asm("add.cc.u32 %0, %0, %3; addc.cc.u32 %1, %1, %4; addc.u32 %2, 0, 0;"
            : "+r"(temp[0]), "+r"(temp[1]), "=r"(temp[2])
            : "r"(a0), "r"(a1));

#if __CUDA_ARCH__ >= 700
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
            : "+r"(temp[0]), "+r"(temp[1])
            : "r"(temp[2]), "r"(gl64_device::W));
#else
        asm("add.cc.u32 %0, %0, %2; addc.u32 %1, %1, 0;"
            : "+r"(temp[0]), "+r"(temp[1])
            : "r"(-temp[2]));
#endif
        asm("mov.b64 %0, {%1, %2};" : "=l"(val) : "r"(temp[0]), "r"(temp[1]));
    }

public:
    friend inline gl64_t operator*(gl64_t a, const uint32_t b)
    {
        a.mul(b);
        a.to();
        return a;
    }
    inline gl64_t &operator*=(const uint32_t a)
    {
        mul(a);
        to();
        return *this;
    }

private:
    inline void reduce()
    {
        uint64_t tmp;
        uint32_t carry;

        asm("add.cc.u64 %0, %2, %3; addc.u32 %1, 0, 0;"
            : "=l"(tmp), "=r"(carry)
            : "l"(val), "l"(0 - MOD));
        asm("{ .reg.pred %top;");
        asm("setp.ne.u32 %top, %0, 0;" ::"r"(carry));
        asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
        asm("}");
    }

public:
#ifdef GL64_PARTIALLY_REDUCED
    inline void to() {}
    inline void from() { reduce(); }
#else
    inline void to() { reduce(); }
    inline void from() {}
#endif

    template <size_t T>
    static inline gl64_t dot_product(const gl64_t a[T], const uint8_t b[T])
    {
        uint32_t lo[2], hi[2], bi;

        bi = b[0];
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(lo[0]), "=r"(lo[1])
            : "r"(a[0].lo()), "r"(bi));
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(hi[0]), "=r"(hi[1])
            : "r"(a[0].hi()), "r"(bi));

        for (uint32_t i = 1; i < T; i++)
        {
            bi = b[i];
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(lo[0]), "+r"(lo[1])
                : "r"(a[i].lo()), "r"(bi));
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(hi[0]), "+r"(hi[1])
                : "r"(a[i].hi()), "r"(bi));
        }

        asm("add.cc.u32 %0, %0, %2; addc.u32 %1, %1, 0;"
            : "+r"(lo[1]), "+r"(hi[1])
            : "r"(hi[0]));

        uint32_t carry;
        asm("sub.cc.u32 %0, 0, %1; subc.u32 %1, %1, 0;"
            : "=r"(hi[0]), "+r"(hi[1]));
        asm("add.cc.u32 %0, %0, %3; addc.cc.u32 %1, %1, %4; addc.u32 %2, 0, 0;"
            : "+r"(lo[0]), "+r"(lo[1]), "=r"(carry)
            : "r"(hi[0]), "r"(hi[1]));
#if __CUDA_ARCH__ >= 700
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
            : "+r"(lo[0]), "+r"(lo[1])
            : "r"(carry), "r"(gl64_device::W));
#else
        asm("add.cc.u32 %0, %0, %2; addc.u32 %1, %1, 0;"
            : "+r"(lo[0]), "+r"(lo[1])
            : "r"(-carry));
#endif
        gl64_t ret;
        asm("mov.b64 %0, {%1, %2};" : "=l"(ret.val) : "r"(lo[0]), "r"(lo[1]));
        ret.to();
        return ret;
    }

    template <size_t T>
    static inline gl64_t dot_product(const gl64_t a[T], const gl64_t b[T])
    {
        uint32_t even[5];
        uint32_t odd[3];

        uint32_t a_lo = a[0].lo(), b_lo = b[0].lo();
        uint32_t a_hi = a[0].hi(), b_hi = b[0].hi();

        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(even[0]), "=r"(even[1])
            : "r"(a_lo), "r"(b_lo));
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(even[2]), "=r"(even[3])
            : "r"(a_hi), "r"(b_hi));
        even[4] = 0;

        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(odd[0]), "=r"(odd[1])
            : "r"(a_lo), "r"(b_hi));
        asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, 0, 0;"
            : "+r"(odd[0]), "+r"(odd[1]), "=r"(odd[2])
            : "r"(a_hi), "r"(b_lo));

        for (uint32_t i = 1; i < T; i++)
        {
            a_lo = a[i].lo(), b_lo = b[i].lo();
            a_hi = a[i].hi(), b_hi = b[i].hi();

            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
                : "+r"(even[0]), "+r"(even[1])
                : "r"(a_lo), "r"(b_lo));
            asm("madc.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, %2, 0;"
                : "+r"(even[2]), "+r"(even[3]), "+r"(even[4])
                : "r"(a_hi), "r"(b_hi));

            asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, %2, 0;"
                : "+r"(odd[0]), "+r"(odd[1]), "+r"(odd[2])
                : "r"(a_hi), "r"(b_lo));
            asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, %2, 0;"
                : "+r"(odd[0]), "+r"(odd[1]), "+r"(odd[2])
                : "r"(a_lo), "r"(b_hi));
        }

        asm("add.cc.u32 %0, %0, %4; addc.cc.u32 %1, %1, %5; addc.cc.u32 %2, %2, %6; addc.u32 %3, %3, 0;"
            : "+r"(even[1]), "+r"(even[2]), "+r"(even[3]), "+r"(even[4])
            : "r"(odd[0]), "r"(odd[1]), "r"(odd[2]));

        // reduce modulo |(mod << 64) + (mod <<32)|
        asm("sub.cc.u32 %0, %0, %3; subc.cc.u32 %1, %1, 0; subc.cc.u32 %2, %2, 0; subc.u32 %3, %3, %3;"
            : "+r"(even[1]), "+r"(even[2]), "+r"(even[3]), "+r"(even[4]));
        asm("sub.u32 %0, %0, %1;"
            : "+r"(even[2])
            : "r"(even[4]));

        gl64_t ret;
        ret.reduce(even);
        ret.to();
        return ret;
    }

private:
    template <int unroll> // 1, 2 or 3
    static __device__ __noinline__ gl64_t sqr_n_mul(gl64_t s, uint32_t n, gl64_t m)
    {
        if (unroll & 1)
        {
            s.mul(s);
            n--;
        }
        if (unroll > 1)
        {
#pragma unroll 1
            do
            {
                s.mul(s);
                s.mul(s);
            } while (n -= 2);
        }
        s.mul(m);

        return s;
    }

public:
    __device__ __noinline__ gl64_t reciprocal() const
    {
        gl64_t t0, t1;

        t1 = sqr_n_mul<1>(*this, 1, *this); // 0b11
        t0 = sqr_n_mul<2>(t1, 2, t1);       // 0b1111
        t0 = sqr_n_mul<2>(t0, 2, t1);       // 0b111111
        t1 = sqr_n_mul<2>(t0, 6, t0);       // 0b111111111111
        t1 = sqr_n_mul<2>(t1, 12, t1);      // 0b111111111111111111111111
        t1 = sqr_n_mul<2>(t1, 6, t0);       // 0b111111111111111111111111111111
        t1 = sqr_n_mul<1>(t1, 1, *this);    // 0b1111111111111111111111111111111
        t1 = sqr_n_mul<2>(t1, 32, t1);      // 0b111111111111111111111111111111101111111111111111111111111111111
        t1 = sqr_n_mul<1>(t1, 1, *this);    // 0b1111111111111111111111111111111011111111111111111111111111111111
        t1.to();

        return t1;
    }
    friend inline gl64_t operator/(int one, const gl64_t &a)
    {
        if (one != 1)
            asm("trap;");
        return a.reciprocal();
    }
    friend inline gl64_t operator/(const gl64_t &a, const gl64_t &b)
    {
        return a * b.reciprocal();
    }
    inline gl64_t &operator/=(const gl64_t &a)
    {
        return *this *= a.reciprocal();
    }

    inline gl64_t heptaroot() const
    {
        gl64_t t0, t1;

        t1 = sqr_n_mul<3>(*this, 3, *this); // 0b1001
        t0 = sqr_n_mul<2>(t1, 6, t1);       // 0b1001001001
        t0 = sqr_n_mul<2>(t0, 12, t0);      // 0b1001001001001001001001
        t0 = sqr_n_mul<2>(t0, 6, t1);       // 0b1001001001001001001001001001
        t1 = sqr_n_mul<2>(t0, 4, *this);    // 0b10010010010010010010010010010001
        t1 = sqr_n_mul<2>(t1, 28, t0);      // 0b100100100100100100100100100100011001001001001001001001001001
        t1 = sqr_n_mul<2>(t1, 2, t0);       // 0b10010010010010010010010010010001101101101101101101101101101101
        t1 = sqr_n_mul<1>(t1, 1, *this);    // 0b100100100100100100100100100100011011011011011011011011011011011
        t1 = sqr_n_mul<1>(t1, 1, *this);    // 0b1001001001001001001001001001000110110110110110110110110110110111
        t1.to();

        return t1;
    }

    static inline void copy(gl64_t &dst, const gl64_t &src)
    {
        dst.val = src.val;
    }

    /* Pack operations */
    static inline void copy_pack(uint64_t nrowsPack, gl64_t *dst, const gl64_t *src)
    {
        for (uint64_t i = 0; i < nrowsPack; ++i)
        {
            dst[i].val = src[i].val;
        }
    }

    static inline void copy_pack(uint64_t nrowsPack, gl64_t *dst, uint64_t stride_dst, const gl64_t *src)
    {
        for (uint64_t i = 0; i < nrowsPack; ++i)
        {
            dst[i * stride_dst].val = src[i].val;
        }
    }

    static inline void op_pack(uint64_t nrowsPack, uint64_t op, gl64_t *c, const gl64_t *a, const gl64_t *b)
    {
        switch (op)
        {
        case 0:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[i] + b[i];
            }
            break;
        case 1:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[i] - b[i];
            }
            break;
        case 2:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = a[i] * b[i];
            }
            break;
        case 3:
            for (uint64_t i = 0; i < nrowsPack; ++i)
            {
                c[i] = b[i] - a[i];
            }
            break;
        default:
            assert(0);
            break;
        }
    }

public:
    static __device__ __forceinline__ void copy_gpu(gl64_t *dst, const gl64_t *src, bool const_src)
    {
        int tid = threadIdx.x;
        if (const_src)
            tid = 0;
        dst[threadIdx.x] = src[tid];
    }

    static __device__ __forceinline__ void copy_gpu(gl64_t *dst, uint64_t stride_dst, const gl64_t *src, bool const_src)
    {
        int tid = threadIdx.x;
        if (const_src)
            tid = 0;
        dst[threadIdx.x * stride_dst] = src[tid];
    }

    static __device__ __forceinline__ void op_gpu(uint64_t op, gl64_t *c, const gl64_t *a, bool const_a, const gl64_t *b, bool const_b)
    {
        int tida = threadIdx.x;
        if (const_a)
            tida = 0;
        int tidb = threadIdx.x;
        if (const_b)
            tidb = 0;

        switch (op)
        {
        case 0:
            c[threadIdx.x] = a[tida] + b[tidb];
            break;
        case 1:
            c[threadIdx.x] = a[tida] - b[tidb];
            break;
        case 2:
            c[threadIdx.x] = a[tida] * b[tidb];
            break;
        case 3:
            c[threadIdx.x] = b[tidb] - a[tida];
            break;
        default:
            assert(0);
            break;
        }
    }
};

struct DeviceCommitBuffers
{
    gl64_t *d_trace;
    gl64_t *d_constPols;
    gl64_t *d_constTree;
    gl64_t *d_aux_trace;
    bool recursive;
};

#undef inline
#undef asm
#endif
#endif
