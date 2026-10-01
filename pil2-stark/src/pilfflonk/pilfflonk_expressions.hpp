#ifndef PILFFLONK_EXPRESSIONS_HPP
#define PILFFLONK_EXPRESSIONS_HPP

#include <cstdint>
#include <vector>

#include "pilfflonk_expressions_bin.hpp"
#include "pilfflonk_info.hpp"

namespace PilFflonk {

// The interpreter of <air>.bin over Fr (pilfflonk/docs/formats.md#bytecode): the case of op 0
// (dim1 = dim1 ∘ dim1) of the STARK's expressions_pack.hpp, with the STARK's operand types
// (OperandTypes) and semantics, and two modes:
//
// - Prover: a code block on every point of a domain (ExpressionsDomain), by blocks of rows run in
//   parallel with OpenMP. The domain is H for the intermediate polynomials, the other expressions
//   and the constraints, and the extended coset g·H' for Q (cExpId).
// - Verifier: a code block at one point ξ, from the evaluations of the evMap, Z_D(ξ) and the
//   scalars. The JS verifier is the real one; this is for tests.
//
// A column at the opening point o = openingPoints[arg2], on point i of a domain of M = 2^e·N points,
// is its value at point (i + 2^e·o) mod M: rows are cyclic.
//
// As in the STARK, an op may write a temporary one of its sources reads: the allocation lets it.
// Unlike the STARK prover's, Zi of lastRow is Z_H(X)/(X − ω^(N−1)), not Z_H(X)/(X − ω^N)
// (pilfflonk/docs/README.md#stark-lastrow-zerofier).
//
// Elements are in ffiasm's Montgomery form. Everything is checked before anything is written: a
// bad argument throws std::invalid_argument; a bytecode that does not fit the AIR, FormatError.

// ω_{2^nBits}: ffiasm's root of unity of order 2^nBits, 5^((r − 1)/2^nBits)
// (pilfflonk/docs/protocol.md#notation). Throws std::invalid_argument for nBits > 28.
FrElement rootOfUnity(uint64_t nBits);

// Zi of each boundary at a point x ∉ H (pilfflonk/docs/protocol.md#constraint-polynomial):
// 1/Z_H(x) for everyRow, and Z_H(x)/Z_D(x) for any other, with Z_H(x) = x^N − 1, Z_D(x) = x − 1
// (firstRow), x − ω^(N−1) (lastRow) and Z_H(x)/Π_j (x − ω^j) over the rows j an everyFrame
// excludes. Throws std::invalid_argument if x is in H.
std::vector<FrElement> zerofiersAt(uint64_t nBits, const std::vector<Boundary> &boundaries, const FrElement &x);

// The rows an everyFrame excludes, ω_N^j for its first offsetMin rows and its last offsetMax, in the
// order of the STARK's buildFrameZerofierInv: its Zi is Π_j (X − ω_N^j). Throws
// std::invalid_argument if they are more than N.
std::vector<FrElement> excludedRoots(uint64_t nBits, const Boundary &boundary);

// The row a firstRow or lastRow boundary excludes: ω_N^0 = 1, or ω_N^(N−1)
// (pilfflonk/docs/protocol.md#constraint-polynomial).
FrElement oneRowRoot(uint64_t nBits, BoundaryType type);

// The points of a part of the extended coset (ExpressionsDomain::cosetPart), c·ω_S^i for i < S =
// 2^partBits, and Z_H on them, which repeats every e = S/N points: what cosetPart computes Zi from,
// and the device's (ExpressionsDomainGpu) too.
struct CosetPart {
    FrElement shift;              // c = g·ω_{N'}^part, g itself for part 0
    FrElement root;               // ω_S
    std::vector<FrElement> zh;    // Z_H(c·ω_S^k) = c^N·ω_e^k − 1 for k < e: at point i, zh[i mod e]
    std::vector<FrElement> zhInv; // 1/zh[k]
};

// Those of part `part` of 2^partBits points of the coset of 2^nBitsExt. Throws as
// ExpressionsDomain::cosetPart, for its arguments and its boundaries, and std::logic_error if Z_H
// vanishes on the coset.
CosetPart cosetPartPoints(uint64_t nBits, uint64_t nBitsExt, uint64_t partBits, uint64_t part,
                          const std::vector<Boundary> &boundaries);

// The points of the prover mode, in their order, and the zerofier terms Zi on them.
class ExpressionsDomain {
public:
    // H: the N = 2^nBits rows of the trace, ω_N^i. It has no Zi: Z_H vanishes on H, so code with a
    // Zi (Q's) cannot run on it.
    static ExpressionsDomain trace(uint64_t nBits);

    // The extended coset g·H', g·ω_{N'}^i for N' = 2^nBitsExt and g = COSET_SHIFT (5), the points
    // of Lde::extendCoset. Zi of each boundary on it, computed as zerofiersAt says, with batch
    // inversions: 32·N' bytes per boundary. Throws std::invalid_argument unless
    // nBits <= nBitsExt <= 28, or for an everyFrame that excludes more than N rows.
    static ExpressionsDomain coset(uint64_t nBits, uint64_t nBitsExt, const std::vector<Boundary> &boundaries);

    // Part `part` of that coset (pilfflonk/docs/protocol.md#q-in-parts), the points of
    // Lde::extendCosetPart: its 2^partBits points g·ω_{N'}^(part + (N'/2^partBits)·i), in that
    // order, and Zi on them, point i the same bit for bit as point part + (N'/2^partBits)·i of
    // coset(). nBits <= partBits <= nBitsExt, and extendBits() is partBits − nBits: a column at
    // opening point o is read o rows later in the part, as on the whole coset. coset() is the one
    // part of partBits = nBitsExt. Throws std::invalid_argument unless
    // nBits <= partBits <= nBitsExt <= 28 and part < N'/2^partBits, or for an everyFrame that
    // excludes more than N rows.
    static ExpressionsDomain cosetPart(uint64_t nBits, uint64_t nBitsExt, uint64_t partBits, uint64_t part,
                                       const std::vector<Boundary> &boundaries);

    uint64_t nBits() const { return nBits_; }
    uint64_t size() const { return uint64_t(1) << (nBits_ + extendBits_); }
    // e: the domain has 2^e points per row of the trace.
    uint64_t extendBits() const { return extendBits_; }
    // The boundaries Zi is defined for: 0 on H.
    uint64_t nZerofiers() const { return zerofiers_.size(); }
    // Zi of boundary b, size() values.
    const std::vector<FrElement> &zerofier(uint64_t b) const { return zerofiers_.at(b); }

private:
    ExpressionsDomain(uint64_t nBits, uint64_t extendBits) : nBits_(nBits), extendBits_(extendBits) {}

    uint64_t nBits_;
    uint64_t extendBits_;
    std::vector<std::vector<FrElement>> zerofiers_;
};

// What the operands of the prover mode read.
struct ProverValues {
    // columns[type][arg1], each with the size() values of the domain in its order: columns[0][id] is
    // fixed column id (constPolsMap), columns[s][stagePos] the committed column of stage s at
    // stagePos (cmPolsMap), for s = 1 … nStages + 1. Only the columns the code reads are needed:
    // the rest may be missing or null.
    std::vector<std::vector<const FrElement *>> columns;
    std::vector<FrElement> publics;
    std::vector<FrElement> challenges; // challengesMap order
    std::vector<FrElement> airValues;
    std::vector<FrElement> proofValues;
    std::vector<FrElement> airgroupValues;
};

// The values of `values` a scalar operand of `type` reads in the prover mode: its publics, air
// values, proof values, airgroup values or challenges. Null for any other type, the numbers too,
// which are the code's (ParserArgs::numbers).
const std::vector<FrElement> *proverScalars(const OperandTypes &types, uint32_t type, const ProverValues &values);

// What the operands of the verifier mode read: no columns, but their evaluations.
struct PointValues {
    std::vector<FrElement> evals;     // evMap order
    std::vector<FrElement> zerofiers; // Zi of each boundary at the point: zerofiersAt
    std::vector<FrElement> publics;
    std::vector<FrElement> challenges;
    std::vector<FrElement> airValues;
    std::vector<FrElement> proofValues;
    std::vector<FrElement> airgroupValues;
};

class Expressions {
public:
    // The code of `bin` for the AIR `info` describes: every column operand's opening point and
    // every Zi's boundary must be the AIR's, and bin's stages its stages, or it throws FormatError.
    // It keeps a reference to `bin`, which must outlive it; of `info` it copies what it needs.
    Expressions(const ExpressionsBin &bin, const PilfflonkInfo &info);

    // Prover mode: expression expId on every point of `domain`, into dest[0 … domain.size()).
    void calculateExpression(uint64_t expId, const ExpressionsDomain &domain, const ProverValues &values,
                             FrElement *dest) const;

    // The same for constraint `index` of section 2: its numerator, 0 on its rows iff it holds.
    void calculateConstraint(uint64_t index, const ExpressionsDomain &domain, const ProverValues &values,
                             FrElement *dest) const;

    // Verifier mode: expression expId at one point.
    FrElement evaluateExpressionAt(uint64_t expId, const PointValues &values) const;

    // The code calculateExpression runs for expId on a domain with nZerofiers Zi, once it has checked
    // what it checks before it computes anything: that there is such an expression, that dest is not
    // null, and every operand the code reads against `values` and the domain. It throws what
    // calculateExpression throws then. For the device's prover mode (ExpressionsGpu), whose values
    // have the same operands.
    const ParserParams &checkedExpression(uint64_t expId, uint64_t nZerofiers, const ProverValues &values,
                                          const void *dest) const;
    // The same of calculateConstraint, for constraint `index` of section 2.
    const ParserParams &checkedConstraint(uint64_t index, uint64_t nZerofiers, const ProverValues &values,
                                          const void *dest) const;

    // Each opening point's shift on a domain of `size` points, 2^extendBits per row of the trace: the
    // prover mode reads a column at openingPoints[k] on point i at point (i + shifts[k]) mod size.
    std::vector<uint64_t> shifts(uint64_t size, uint64_t extendBits) const;

    // Rows per block of the prover mode, the STARK's NROWS_PACK (fewer if the domain is smaller).
    static constexpr uint64_t BLOCK_ROWS = 128;

private:
    void checkOperands(const ParserParams &params, const ParserArgs &args, uint64_t nZerofiers,
                       const ProverValues &values, const void *dest, const char *what) const;
    void calculate(const ParserParams &params, const ParserArgs &args, const ExpressionsDomain &domain,
                   const ProverValues &values, FrElement *dest) const;

    const ExpressionsBin &bin;
    std::vector<int64_t> openingPoints;
};

} // namespace PilFflonk

#endif
