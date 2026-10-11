#ifndef PIL2_FFLONK_FR_PACK_HPP
#define PIL2_FFLONK_FR_PACK_HPP

#include <cstdint>
#include <alt_bn128.hpp>

// The field-operation primitive the BN254 expression evaluator is built on --
// the counterpart of Goldilocks::op_pack in goldilocks_base_field_pack.hpp.
//
// pil2's expression bytecode is field agnostic: `args` carries opcodes and
// operand indices, never field elements, so the same `.bin` drives Goldilocks
// and BN254 alike. What differs is the arithmetic underneath and, crucially,
// the *dimension* handling.
//
// The Goldilocks interpreter dispatches three dimension combinations --
//   case 0: dim1 (op) dim1  -> Goldilocks::op_pack
//   case 1: dim3 (op) dim1  -> Goldilocks3::op_31_pack
//   case 2: dim3 (op) dim3  -> Goldilocks3::op_pack
// because its challenges live in a cubic extension. BN254's scalar field is
// already 254 bits, so challenges live in Fr directly and every operand is
// dim 1: only case 0 exists here, and the whole Goldilocks3 half of the
// evaluator disappears. This was checked against pil-fflonk's generated
// chelpers, which contain 477 FrElement operations and no extension arithmetic
// whatsoever.
//
// NROWS_PACK is retained from the Goldilocks evaluator, but it means something
// different: there it also enables SIMD over 64-bit elements, whereas an Fr
// element is four limbs and a multiplication is a multi-limb asm routine, so
// blocking here buys cache locality and loop-overhead amortisation only.

namespace PilFflonk {

// Deliberately not aliased to `FrElement`: ffiasm already has a global type of
// that name -- the tagged C API struct -- and shadowing it is exactly the
// confusion documented above. The engine's element type is spelled out.
using FrEl = typename AltBn128::Engine::FrElement;

// Matches Goldilocks' NROWS_PACK so the two evaluators block identically.
constexpr uint64_t NROWS_PACK = 128;

// Opcodes, in the order the bytecode encodes them. Must not be reordered:
// these values come from the generated `.bin`, not from this header.
enum class FrOp : uint64_t {
    add = 0,      // c = a + b
    sub = 1,      // c = a - b
    mul = 2,      // c = a * b
    sub_rev = 3,  // c = b - a
};

// c[i] = a[i] (op) b[i] for i in [0, nrowsPack).
inline void op_pack(AltBn128::Engine &E, uint64_t nrowsPack, uint64_t op, FrEl *c, const FrEl *a,
                    const FrEl *b) {
    switch (op) {
        case 0:
            for (uint64_t i = 0; i < nrowsPack; ++i) c[i] = E.fr.add(a[i], b[i]);
            break;
        case 1:
            for (uint64_t i = 0; i < nrowsPack; ++i) c[i] = E.fr.sub(a[i], b[i]);
            break;
        case 2:
            for (uint64_t i = 0; i < nrowsPack; ++i) c[i] = E.fr.mul(a[i], b[i]);
            break;
        case 3:
            for (uint64_t i = 0; i < nrowsPack; ++i) c[i] = E.fr.sub(b[i], a[i]);
            break;
        default:
            throw std::runtime_error("PilFflonk::op_pack: unknown op " + std::to_string(op));
    }
}

// As above, but either operand may be a single value broadcast across the
// block. The evaluator marks an operand constant when its index lies past the
// commit buffers, so this is the common path for challenges and constants.
inline void op_pack(AltBn128::Engine &E, uint64_t nrowsPack, uint64_t op, FrEl *c, const FrEl *a,
                    const bool const_a, const FrEl *b, const bool const_b) {
    if (const_a && const_b) {
        FrEl r;
        switch (op) {
            case 0: r = E.fr.add(a[0], b[0]); break;
            case 1: r = E.fr.sub(a[0], b[0]); break;
            case 2: r = E.fr.mul(a[0], b[0]); break;
            case 3: r = E.fr.sub(b[0], a[0]); break;
            default: throw std::runtime_error("PilFflonk::op_pack: unknown op " + std::to_string(op));
        }
        for (uint64_t i = 0; i < nrowsPack; ++i) c[i] = r;
        return;
    }

    if (const_a) {
        switch (op) {
            case 0: for (uint64_t i = 0; i < nrowsPack; ++i) c[i] = E.fr.add(a[0], b[i]); break;
            case 1: for (uint64_t i = 0; i < nrowsPack; ++i) c[i] = E.fr.sub(a[0], b[i]); break;
            case 2: for (uint64_t i = 0; i < nrowsPack; ++i) c[i] = E.fr.mul(a[0], b[i]); break;
            case 3: for (uint64_t i = 0; i < nrowsPack; ++i) c[i] = E.fr.sub(b[i], a[0]); break;
            default: throw std::runtime_error("PilFflonk::op_pack: unknown op " + std::to_string(op));
        }
        return;
    }

    if (const_b) {
        switch (op) {
            case 0: for (uint64_t i = 0; i < nrowsPack; ++i) c[i] = E.fr.add(a[i], b[0]); break;
            case 1: for (uint64_t i = 0; i < nrowsPack; ++i) c[i] = E.fr.sub(a[i], b[0]); break;
            case 2: for (uint64_t i = 0; i < nrowsPack; ++i) c[i] = E.fr.mul(a[i], b[0]); break;
            case 3: for (uint64_t i = 0; i < nrowsPack; ++i) c[i] = E.fr.sub(b[0], a[i]); break;
            default: throw std::runtime_error("PilFflonk::op_pack: unknown op " + std::to_string(op));
        }
        return;
    }

    op_pack(E, nrowsPack, op, c, a, b);
}

} // namespace PilFflonk

#endif // PIL2_FFLONK_FR_PACK_HPP
