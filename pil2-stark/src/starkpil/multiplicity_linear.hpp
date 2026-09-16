#ifndef MULTIPLICITY_LINEAR_HPP
#define MULTIPLICITY_LINEAR_HPP

#include <cstdint>
#include "multiplicity_decoders.hpp"

// A lookup reduced to `value = sum(coef[i] * col[i]) + konst` (all arithmetic in Goldilocks), which
// a dedicated kernel evaluates in registers instead of round-tripping through `computeExpressions_`
// and its global scratch. The std range checks are linear by construction; anything that does not
// reduce to this shape is left `ok = false` and keeps the interpreter, so the extractor may be
// conservative but never optimistic.
#define MUL_MAX_TERMS 8

// Coefficients are canonical field elements, not signed integers: a `numbers` constant can be any
// element of [0, p), which would wrap an int64 on the first negation.
inline uint64_t mulNegFE(uint64_t a) { return a == 0 ? 0 : MUL_P - a; }
inline uint64_t mulSubFE(uint64_t a, uint64_t b) { return mulAddFE(a, mulNegFE(b)); }

// `type` is the operand type as load__ decodes it: below bufferCommitSize a pol buffer (0 const,
// 1.. the committed stages), at or above it one of the uniform pools. A uniform operand is only
// known once the proof starts, so it stays a term with no row dependence rather than folding away.
struct MulLinTerm {
    uint16_t type   = 0;
    uint16_t argIdx = 0;   // column id within that buffer
    uint16_t rowOff = 0;   // index into nextStridesExps, for `'`-shifted references
    uint64_t coef   = 1;
};

// `A`, or `A * B` when a product of two varying values appears -- which is what a selector built
// from a couple of flags compiles to. Kept to two factors on purpose: that covers every selector in
// the PIL today, and anything deeper stays `ok = false` rather than being approximated.
struct MulLinForm {
    uint8_t     n     = 0;
    bool        ok    = false;
    uint64_t    konst = 0;
    MulLinTerm  t[MUL_MAX_TERMS];
    // The second factor. `hasProduct` is false for a plain linear form.
    bool        hasProduct = false;
    uint8_t     n2    = 0;
    uint64_t    konst2 = 0;
    MulLinTerm  t2[MUL_MAX_TERMS];
};

// A SUM of products: `sum_i p[i]`, each p[i] being `A` or `A * B`.
//
// The extractor used to refuse `X + Y*Z` outright ("add over a product"), which is the shape
// Keccak's xor5 tuple compiles to -- and, more broadly, 30 of the 41 lookups that still fall
// through to the interpreter. One product was enough for a selector built from two flags; a tuple
// element that mixes a product into a sum needs a sum.
#define MUL_MAX_PRODUCTS 3

// How many summands the walker actually needed when it last overflowed, so the cap can be set from
// a measurement instead of a guess. Counting continues past the limit; only the storing stops.
inline uint32_t& mulProductsNeeded() { static uint32_t n = 0; return n; }

struct MulPoly {
    bool       ok    = false;
    uint8_t    nProd = 0;
    MulLinForm p[MUL_MAX_PRODUCTS];
};

// Drop terms whose coefficient cancelled to zero.
inline void mulLinCompact(MulLinForm& f) {
    uint8_t w = 0;
    for (uint8_t i = 0; i < f.n; ++i) if (f.t[i].coef != 0) f.t[w++] = f.t[i];
    f.n = w;
}

inline bool mulLinAddTerm(MulLinForm& dst, MulLinTerm term) {
    // Fold repeated references to one column so the kernel loads it once.
    for (uint8_t j = 0; j < dst.n; ++j) {
        if (dst.t[j].type == term.type && dst.t[j].argIdx == term.argIdx
            && dst.t[j].rowOff == term.rowOff) {
            dst.t[j].coef = mulAddFE(dst.t[j].coef, term.coef);
            return true;
        }
    }
    if (dst.n >= MUL_MAX_TERMS) return false;
    dst.t[dst.n++] = term;
    return true;
}

// dst = a + b, or a - b when negate.
inline bool mulLinAdd(MulLinForm& dst, const MulLinForm& a, const MulLinForm& b, bool negate) {
    MulLinForm r{};
    r.konst = negate ? mulSubFE(a.konst, b.konst) : mulAddFE(a.konst, b.konst);
    for (uint8_t i = 0; i < a.n; ++i) if (!mulLinAddTerm(r, a.t[i])) return false;
    for (uint8_t i = 0; i < b.n; ++i) {
        MulLinTerm term = b.t[i];
        if (negate) term.coef = mulNegFE(term.coef);
        if (!mulLinAddTerm(r, term)) return false;
    }
    mulLinCompact(r);
    r.ok = true;
    dst = r;
    return true;
}

// Scaling stays linear only when one side is a constant.
inline bool mulLinScale(MulLinForm& dst, const MulLinForm& a, uint64_t k) {
    MulLinForm r = a;
    r.konst = mulMulFE(a.konst, k);
    for (uint8_t i = 0; i < r.n; ++i) r.t[i].coef = mulMulFE(r.t[i].coef, k);
    mulLinCompact(r);
    r.ok = true;
    dst = r;
    return true;
}

inline MulLinForm mulLinConst(uint64_t k) {
    MulLinForm f{};
    f.ok = true;
    f.konst = k;
    return f;
}

// The uniform pools, in load__'s operand-type order. Challenges and evals are absent on purpose:
// they do not exist when the witness commits, which is when the scatter runs.
inline bool mulIsUniformType(uint16_t type, uint32_t base) {
    return type == base + 2 || type == base + 4 || type == base + 5 || type == base + 6;
}

inline MulLinForm mulLinColumn(uint16_t type, uint16_t argIdx, uint16_t rowOff) {
    MulLinForm f{};
    f.ok = true;
    f.n = 1;
    f.t[0] = MulLinTerm{ type, argIdx, rowOff, 1 };
    return f;
}

// ---- sum-of-products helpers -----------------------------------------------------------------
inline MulPoly mulPolyOf(const MulLinForm& f) {
    MulPoly q{};
    if (!f.ok) return q;
    q.ok = true;
    q.nProd = 1;
    q.p[0] = f;
    return q;
}

inline bool mulPolyIsConst(const MulPoly& a) {
    return a.nProd == 1 && a.p[0].n == 0 && !a.p[0].hasProduct;
}

// Concatenate, folding a plain-linear summand into an existing plain-linear one so the common
// `a + b + c` case still costs a single product slot.
inline bool mulPolyAdd(MulPoly& dst, const MulPoly& a, const MulPoly& b, bool negate) {
    uint32_t overflow = 0;
    MulPoly r = a;
    for (uint8_t i = 0; i < b.nProd; ++i) {
        MulLinForm term = b.p[i];
        if (negate) {
            MulLinForm neg{};
            if (!mulLinScale(neg, term, mulNegFE(1))) return false;
            term = neg;
        }
        bool merged = false;
        if (!term.hasProduct) {
            for (uint8_t j = 0; j < r.nProd; ++j) {
                if (r.p[j].hasProduct) continue;
                MulLinForm sum{};
                if (mulLinAdd(sum, r.p[j], term, false)) { r.p[j] = sum; merged = true; }
                break;
            }
        }
        if (merged) continue;
        if (r.nProd >= MUL_MAX_PRODUCTS) { ++overflow; continue; }
        r.p[r.nProd++] = term;
    }
    if (overflow != 0) {
        // only meaningful for the call that overflowed
        mulProductsNeeded() = (uint32_t)r.nProd + overflow;
        return false;
    }
    r.ok = true;
    dst = r;
    return true;
}

inline bool mulPolyScale(MulPoly& dst, const MulPoly& a, uint64_t k) {
    MulPoly r = a;
    for (uint8_t i = 0; i < r.nProd; ++i)
        if (!mulLinScale(r.p[i], a.p[i], k)) return false;
    r.ok = true;
    dst = r;
    return true;
}

// Only shapes that stay inside the representation: constant * anything, or linear * linear.
inline bool mulPolyMul(MulPoly& dst, const MulPoly& a, const MulPoly& b) {
    if (mulPolyIsConst(a)) return mulPolyScale(dst, b, a.p[0].konst);
    if (mulPolyIsConst(b)) return mulPolyScale(dst, a, b.p[0].konst);
    if (a.nProd != 1 || b.nProd != 1 || a.p[0].hasProduct || b.p[0].hasProduct) return false;
    MulLinForm r = a.p[0];
    r.hasProduct = true;
    r.n2 = b.p[0].n;
    r.konst2 = b.p[0].konst;
    for (uint8_t q = 0; q < b.p[0].n; ++q) r.t2[q] = b.p[0].t[q];
    r.ok = true;
    dst = mulPolyOf(r);
    return true;
}


#endif
