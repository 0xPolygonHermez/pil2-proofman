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

#endif
