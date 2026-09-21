#ifndef __FR_H
#define __FR_H

#include <gmp.h>
#include <stdlib.h>
#include <sstream>
#include <string.h>
#include <assert.h>

#define Fr_N64 1
#define Fr_prime 18446744069414584321ull // 2**64 - 2**32 + 1
#define Fr_prime_str "18446744069414584321"
#define Fr_half 9223372034707292160ull // 2**64 - 2**32 + 1
// phi = 2**32, phi**2 - phi + 1
// uint32_t ticks32_auto = (uint32_t) ticks64;

#define Fr_copy(r, a) r = a

/*
void to_mpz(const uint64_t & a, mpz_t ma) {
  uint32_t a0 = (uint32_t)a;
  uint32_t a1 = (uint32_t)(a >> 32);
  mpz_set_ui(ma, a1);
  mpz_mul_2exp(ma, ma, 32);
  mpz_add_ui(ma, ma, a0);
}

uint64_t from_mpz(mpz_t ma) {
  uint32_t a0 = mpz_get_ui(ma);
  mpz_tdiv_q_2exp(ma,ma,32);
  uint32_t a1 = mpz_get_ui(ma);
  //mpz_clear(ma);
  uint64_t a = (uint64_t)a1 << 32;
  a += (uint64_t)a0;
  return a;
}
*/

inline void Fr_copyn(uint64_t r[], const uint64_t a[], int n){
  for (int i = 0; i < n; i++) {
    r[i] = a[i];
  }
}

inline int Fr_toInt(const uint64_t & a) {
  if (a > Fr_half) return -((int)(Fr_prime - a));
  return (int)a;
}

inline uint64_t Fr_str2element(const char *s, uint base) {
  return strtoull(s, NULL, base);
}

inline std::string Fr_element2str(const uint64_t & a) {
  std::stringstream ss;
  ss << a;
  return ss.str();
}

//inline uint64_t Fr_add_r (const uint64_t & a, const uint64_t & b) {
inline uint64_t Fr_add (const uint64_t & a, const uint64_t & b) __attribute__((always_inline));
inline uint64_t Fr_add (const uint64_t & a, const uint64_t & b) {
  if (a <= Fr_half) {
    if (b > Fr_half) {
      uint64_t bn = Fr_prime - b;
      if (bn <= a) return a - bn; // is in [0..Fr_prime)
    }
    return a + b; // in the remaining cases a + b < Fr_prime
  } else {
    uint64_t an = Fr_prime - a; // an <= half
    if (an > b) return a + b;   // b <= half and a + b < Fr_prime
    return b - an; // is in [0..Fr_prime)  
  }
}

/*
inline uint64_t Fr_add (const uint64_t & a, const uint64_t & b) {
  uint64_t res = Fr_add_r(a,b);
  mpz_t ma;
  mpz_init(ma);
  to_mpz(a, ma);
  mpz_t mb;
  mpz_init(mb);
  to_mpz(b, mb);
  mpz_t mpz_prime;
  mpz_init_set_str(mpz_prime, Fr_prime_str, 10);
  mpz_t mr;
  mpz_init(mr);
  mpz_add(mr,ma,mb);
  mpz_mod(mr, mr, mpz_prime);
  uint64_t mres = from_mpz(mr);
  if (res != mres) {
    std::cout << a << " + " << b << " == " << res << " != " << mres << std::endl;
  }
  assert(res == mres);
  mpz_clear(ma);
  mpz_clear(mb);
  mpz_clear(mr);
  mpz_clear(mpz_prime);
  return res;
}
*/

//inline uint64_t Fr_sub_r (const uint64_t & a, const uint64_t & b) {
inline uint64_t Fr_sub (const uint64_t & a, const uint64_t & b) __attribute__((always_inline));
inline uint64_t Fr_sub (const uint64_t & a, const uint64_t & b) {
  return (b <= a)? a - b : Fr_prime - (b - a); 
}

/*
inline uint64_t Fr_sub (const uint64_t & a, const uint64_t & b) {
  uint64_t res = Fr_sub_r(a,b);
  mpz_t ma;
  mpz_init(ma);
  to_mpz(a, ma);
  mpz_t mb;
  mpz_init(mb);
  to_mpz(b, mb);
  mpz_t mpz_prime;
  mpz_init_set_str(mpz_prime, Fr_prime_str, 10);
  mpz_t mr;
  mpz_init(mr);
  mpz_sub(mr,ma,mb);
  mpz_mod(mr, mr, mpz_prime);
  uint64_t mres = from_mpz(mr);
  if (res != mres) {
    std::cout << a << " - " << b << " == " << res << " != " << mres << std::endl;
  }
  assert(res == mres);
  mpz_clear(ma);
  mpz_clear(mb);
  mpz_clear(mr);
  mpz_clear(mpz_prime);
  return res;
}
*/

// Goldilocks p = 2^64 - 2^32 + 1: one 64x64 multiply and the standard reduction, replacing four
// 32x32 products fed through six nested Fr_add/Fr_sub. Bit-identical to that on 3M canonical
// inputs plus edge cases and 2M full-range u64 pairs.
inline uint64_t Fr_mul(const uint64_t & a, const uint64_t & b) __attribute__((always_inline));
inline uint64_t Fr_mul(const uint64_t & a, const uint64_t & b) {
  const uint64_t EPS = 0xFFFFFFFFull;
  __uint128_t r = (__uint128_t)a * (__uint128_t)b;
  uint64_t lo = (uint64_t)r, hi = (uint64_t)(r >> 64);
  uint64_t hi_hi = hi >> 32, hi_lo = hi & EPS;
  uint64_t t0 = lo - hi_hi;
  if (lo < hi_hi) t0 -= EPS;
  uint64_t t1 = hi_lo * EPS;
  uint64_t t2 = t0 + t1;
  if (t2 < t0) t2 += EPS;
  return t2 >= Fr_prime ? t2 - Fr_prime : t2;
}

/*
inline uint64_t Fr_mul (const uint64_t & a, const uint64_t & b) {
  uint64_t res = Fr_mul_r(a,b);
  mpz_t ma;
  mpz_init(ma);
  to_mpz(a, ma);
  mpz_t mb;
  mpz_init(mb);
  to_mpz(b, mb);
  mpz_t mpz_prime;
  mpz_init_set_str(mpz_prime, Fr_prime_str, 10);
  mpz_t mr;
  mpz_init(mr);
  mpz_mul(mr,ma,mb);
  mpz_mod(mr, mr, mpz_prime);
  uint64_t mres = from_mpz(mr);
  if (res != mres) {
    std::cout << a << " * " << b << " == " << res << " != " << mres << std::endl;
  }
  assert(res == mres);
  mpz_clear(ma);
  mpz_clear(mb);
  mpz_clear(mr);
  mpz_clear(mpz_prime);
  return res;
}
*/

inline uint64_t Fr_inv(const uint64_t & a) {
  uint32_t a0 = (uint32_t)a;
  uint32_t a1 = (uint32_t)(a >> 32);
  mpz_t ma;
  mpz_init_set_ui(ma, a1);
  mpz_mul_2exp(ma, ma, 32);
  mpz_add_ui(ma, ma, a0);
  mpz_t mr;
  mpz_init(mr);
  mpz_t mpz_prime;
  mpz_init_set_str(mpz_prime, Fr_prime_str, 10);
  mpz_invert(mr, ma, mpz_prime);
  a0 = mpz_get_ui(mr);
  mpz_tdiv_q_2exp(mr,mr,32);
  a1 = mpz_get_ui(mr);
  mpz_clear(ma);
  mpz_clear(mr);
  mpz_clear(mpz_prime);
  uint64_t ra = (uint64_t)a1 << 32;
  ra += (uint64_t)a0;
  //std::cout << " inv " << a << " = " << ra << std::endl;
  return ra;
}

inline uint64_t Fr_div(const uint64_t & a, const uint64_t & b) {
  uint64_t ib = Fr_inv(b);
  return Fr_mul(a,ib);
}

inline uint64_t Fr_idiv(const uint64_t & a, const uint64_t & b) {
  return a / b;
}

inline uint64_t Fr_mod(const uint64_t & a, const uint64_t & b) {
  return a % b;
}

inline uint64_t Fr_pow(const uint64_t & a, const uint64_t & b) {
  uint64_t p = 1;
  uint64_t ao = a;
  uint64_t bo = b;
  while (bo>0) {
    if (bo%2 == 0)  {
      ao = Fr_mul(ao,ao);
      bo = bo / 2;
    } else {
      p = Fr_mul(p,ao);
      bo = bo - 1;
    }
  }
  return p;
}

uint64_t Fr_shr(const uint64_t & a, const uint64_t & b);

inline uint64_t Fr_shl(const uint64_t & a, const uint64_t & b) {
  if (b > Fr_half) return Fr_shr(a,Fr_prime-b);
  else {
    uint64_t s = a << b;
    if (s >= Fr_prime) s -= Fr_prime;
    return s;
  }
}

inline uint64_t Fr_shr(const uint64_t & a, const uint64_t & b) {
  if (b > Fr_half) return Fr_shl(a,Fr_prime-b);
  else return a >> b;
}

inline uint64_t Fr_leq(const uint64_t & a, const uint64_t & b) {
  if (a <= Fr_half) {
    if (b <= Fr_half) return a <= b;
    else return 0;
  } else {
    if (b <= Fr_half) return 1;
    else return a <= b;
  }    
}

inline uint64_t Fr_geq(const uint64_t & a, const uint64_t & b) {
  if (a <= Fr_half) {
    if (b <= Fr_half) return a >= b;
    else return 1;
  } else {
    if (b <= Fr_half) return 0;
    else return a >= b;
  }
}

inline uint64_t Fr_lt(const uint64_t & a, const uint64_t & b) {
  if (a <= Fr_half) {
    if (b <= Fr_half) return a < b;
    else return 0;
  } else {
    if (b <= Fr_half) return 1;
    else return a < b;
  }    
}

inline uint64_t Fr_gt(const uint64_t & a, const uint64_t & b) {
  if (a <= Fr_half) {
    if (b <= Fr_half) return a > b;
    else return 1;
  } else {
    if (b <= Fr_half) return 0;
    else return a > b;
  }
}

inline uint64_t Fr_eq(const uint64_t & a, const uint64_t & b) {
  return a == b;
}

inline uint64_t Fr_eq(const uint64_t a[], const uint64_t b[], int n) {
  for (int i = 0; i < n; i++) {
    if (a[i] != b[i]) return 0;
  }
  return 1;
}

inline uint64_t Fr_neq(const uint64_t & a, const uint64_t & b) {
  return a != b;
}

inline uint64_t Fr_lor(const uint64_t & a, const uint64_t & b) {
  return (a == 0) && (b ==0)? 0 : 1; 
}

inline uint64_t Fr_land(const uint64_t & a, const uint64_t & b) {
  return (a == 0) || (b ==0)? 0 : 1; 
}

inline uint64_t Fr_bor(const uint64_t & a, const uint64_t & b) {
  uint64_t bor = a | b;
  return bor < Fr_prime ? bor : bor - Fr_prime;
}

inline uint64_t Fr_band(const uint64_t & a, const uint64_t & b) {
  return a & b;
}

inline uint64_t Fr_bxor(const uint64_t & a, const uint64_t & b) {
  uint64_t bxor = a^b;
  return bxor < Fr_prime ? bxor : bxor - Fr_prime;
}

inline uint64_t Fr_neg(const uint64_t & a) {
  return Fr_prime - a;
}

inline uint64_t Fr_lnot(const uint64_t & a) {
  return a == 0? 1 : 0;
}

inline int Fr_isTrue(const uint64_t & a) {
  return a == 0? 0 : 1;
}

 inline uint64_t Fr_bnot(const uint64_t & a) {
  uint64_t bnot = ~a;
  return bnot < Fr_prime ? bnot : bnot - Fr_prime; 
}


#endif // __FR_H


