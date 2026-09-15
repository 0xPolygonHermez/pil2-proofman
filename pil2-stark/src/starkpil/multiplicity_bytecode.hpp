#ifndef MULTIPLICITY_BYTECODE_HPP
#define MULTIPLICITY_BYTECODE_HPP

#include <cstdint>
#include <vector>
#include "multiplicity_linear.hpp"

// The bytecode walk, with no dependency on SetupCtx so it can be exercised directly by tests.
//
// Layout, mirroring the `ops[kk] == 0` case of computeExpressions_:
//   ops[kk]        dimension case; 0 is dim1-op-dim1, the only shape a linear expression produces
//   args[i+0]      arithmetic op: 0 add, 1 sub, 2 mul, 3 rsub
//   args[i+1]      destination temp index
//   args[i+2..4]   src0 as (type, argIdx, argOffset)
//   args[i+5..7]   src1
// `type` decodes as in load__: at or below `nSections` it names a pol buffer (0 const, 1.. the
// committed stages), `base` and `base+1` are the dim1/dim3 temporaries, and `base+2` upwards are
// the constant pools -- of which only `base+3` (numbers) is known before the proof starts.

struct MulByteCode {
    const uint8_t*  ops     = nullptr;
    const uint16_t* args    = nullptr;
    const uint64_t* numbers = nullptr;   // canonical field elements
    uint32_t nOps      = 0;
    uint32_t nTemp1    = 0;
    uint32_t base      = 0;              // bufferCommitSize
    uint32_t nSections = 0;              // highest valid pol-buffer type
    uint32_t customBase = 0;             // first custom-commit operand type (nStages + 4)
    uint32_t nCustom = 0;
};

// Conservative by construction: anything the walk does not fully understand returns `ok = false`,
// and the caller keeps the interpreter for that expression.
// Why the last walk gave up. A single global is enough: the plan is built once, single-threaded,
// and this is only read straight after a failure.
inline const char*& mulWalkFailReason() { static const char* r = nullptr; return r; }

inline MulLinForm mulWalkBytecode(const MulByteCode& bc) {
    mulWalkFailReason() = nullptr;
    auto bail = [&](const char* why) { mulWalkFailReason() = why; return MulLinForm{}; };
    const MulLinForm bad{};
    if (bc.nOps == 0 || bc.ops == nullptr || bc.args == nullptr) return bail("no bytecode");
    std::vector<MulLinForm> tmp(bc.nTemp1 + 1);

    auto operand = [&](uint16_t type, uint16_t argIdx, uint16_t argOff, MulLinForm& out) -> bool {
        if (type == bc.base) {                                    // dim1 temporary
            if (argIdx >= tmp.size() || !tmp[argIdx].ok) { mulWalkFailReason() = "unresolved temporary"; return false; }
            out = tmp[argIdx];
            return true;
        }
        if (type == bc.base + 1) { mulWalkFailReason() = "dim3 temporary"; return false; }                    // dim3 temporary
        if (mulIsUniformType(type, bc.base)) {                    // publics, air/proof/airgroup values
            out = mulLinColumn(type, argIdx, 0);
            return true;
        }
        if (type == bc.base + 3 && bc.numbers != nullptr) {       // numbers pool
            out = mulLinConst(mulCanonHD(bc.numbers[argIdx]));
            return true;
        }
        if (type >= bc.base + 2) { mulWalkFailReason() = "challenge/eval operand"; return false; }                    // challenge / eval
        if (type > bc.nSections) return false;                    // zi / xDivXSub
        out = mulLinColumn(type, argIdx, argOff);
        return true;
    };

    uint64_t i = 0;
    MulLinForm last{};
    for (uint32_t k = 0; k < bc.nOps; ++k) {
        if (bc.ops[k] != 0) return bail("non dim1-op-dim1 operation");                       // not dim1-op-dim1
        MulLinForm a, b, r;
        if (!operand(bc.args[i + 2], bc.args[i + 3], bc.args[i + 4], a))
            return bail(mulWalkFailReason() ? mulWalkFailReason() : "src0");
        if (!operand(bc.args[i + 5], bc.args[i + 6], bc.args[i + 7], b))
            return bail(mulWalkFailReason() ? mulWalkFailReason() : "src1");

        switch (bc.args[i]) {
            case 0: if (a.hasProduct || b.hasProduct) return bail("add over a product");
                    if (!mulLinAdd(r, a, b, false)) return bail("too many terms"); break;   // a + b
            case 1: if (a.hasProduct || b.hasProduct) return bail("sub over a product");
                    if (!mulLinAdd(r, a, b, true))  return bail("too many terms"); break;   // a - b
            case 3: if (a.hasProduct || b.hasProduct) return bail("rsub over a product");
                    if (!mulLinAdd(r, b, a, true))  return bail("too many terms"); break;   // b - a
            case 2:
                if      (a.n == 0 && !a.hasProduct) { if (!mulLinScale(r, b, a.konst)) return bail("too many terms"); }
                else if (b.n == 0 && !b.hasProduct) { if (!mulLinScale(r, a, b.konst)) return bail("too many terms"); }
                else if (!a.hasProduct && !b.hasProduct) {
                    // Two varying factors: keep them as a product rather than giving up. Any
                    // further arithmetic on the result is refused below, so this cannot silently
                    // approximate a deeper expression.
                    r = a;
                    r.hasProduct = true;
                    r.n2 = b.n;
                    r.konst2 = b.konst;
                    for (uint8_t q = 0; q < b.n; ++q) r.t2[q] = b.t[q];
                    r.ok = true;
                } else return bail("product of products");
                break;
            default: return bail("unknown arithmetic op");
        }
        if (k + 1 == bc.nOps) last = r;
        else {
            const uint16_t dst = bc.args[i + 1];
            if (dst >= tmp.size()) return bail("temp index out of range");
            tmp[dst] = r;
        }
        i += 8;
    }
    return last;
}

#endif
