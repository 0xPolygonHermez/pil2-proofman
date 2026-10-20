#ifndef PILFFLONK_EXPRESSIONS_KERNELS_HPP
#define PILFFLONK_EXPRESSIONS_KERNELS_HPP

#include <cstddef>
#include <cstdint>

// The kernels of the prover mode of the interpreter on the device (pilfflonk_expressions.cu), with
// C linkage as pilfflonk_kernels.hpp's, and the tables their operands are read from, which the host
// builds (ExpressionsGpu, pilfflonk_expressions_gpu.hpp) and the kernels and the host's tests read
// through operandAddress. Pointers are device memory unless said otherwise; an element is a BN128
// scalar of 32 bytes in Montgomery form; the kernels run on the legacy default stream, return before
// they finish, and abort the process on a CUDA failure (CHECKCUDAERR).

#if defined(__CUDACC__)
#define PILFFLONK_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define PILFFLONK_HOST_DEVICE inline
#endif

namespace PilFflonk {

// The bytes of an element.
constexpr uint64_t OPERAND_BYTES = 32;

// The rows a block of the interpreter's kernel evaluates at a time, one per thread: the CPU's
// Expressions::BLOCK_ROWS, the STARK's NROWS_PACK. Its temporaries are nTemp of them per thread.
constexpr uint32_t EXPRESSION_ROWS = 128;

// The ops of the bytecode, args[0] of an op (OpCode of pilfflonk_expressions_bin.hpp, which
// pilfflonk_expressions_gpu.cpp checks they are).
enum ExpressionOp : uint32_t { EXPRESSION_ADD = 0, EXPRESSION_SUB = 1, EXPRESSION_MUL = 2, EXPRESSION_SUB_SWAP = 3 };

// Where the operands of one evaluation of a code block are, for any operand but a temporary
// (OperandTypes, pilfflonk_expressions_bin.hpp): every table is in one piece of memory the host
// fills (encodeOperands).
struct OperandTables {
    const void *const *columns;   // column (type, arg1) is columns[columnStart[type] + arg1]
    const uint32_t *columnStart;  // by type, 0 … ziType − 1
    const uint64_t *shifts;       // by opening point: the column is read shifts[arg2] points later
    const void *const *zerofiers; // by boundary arg1 − 1, Zi at point i at zerofiers[b][i & zerofierMasks[b]]
    const uint64_t *zerofierMasks;
    const void *const *scalars;   // by type − scalarsType: publics, numbers, air values, proof values,
                                  // airgroup values and challenges, their arg1-th element
    uint64_t mask;                // the domain's size − 1: columns are read cyclically
    uint32_t ziType;              // OperandTypes::zi(): the columns are the types below it
    uint32_t scalarsType;         // OperandTypes::publics(): the scalars are the types from it
};

// The address of operand (type, arg1, arg2) at point i, of any type but a temporary's.
PILFFLONK_HOST_DEVICE const void *operandAddress(const OperandTables &t, uint32_t type, uint32_t arg1, uint32_t arg2,
                                                 uint64_t i) {
    const void *base;
    uint64_t index;
    if (type < t.ziType) {
        base = t.columns[t.columnStart[type] + arg1];
        index = (i + t.shifts[arg2]) & t.mask;
    } else if (type == t.ziType) {
        base = t.zerofiers[arg1 - 1];
        index = i & t.zerofierMasks[arg1 - 1];
    } else {
        base = t.scalars[type - t.scalarsType];
        index = arg1;
    }
    return static_cast<const uint8_t *>(base) + index * OPERAND_BYTES;
}

// One evaluation of a code block of the bytecode on the `size` points of a domain.
struct ExpressionLaunch {
    const uint32_t *args; // the code: nOps ops of 8 args, 16-byte aligned
    uint32_t nOps;
    uint32_t nTemp;
    uint32_t tmpType; // OperandTypes::tmp()
    uint32_t blocks;  // of EXPRESSION_ROWS threads, each block striding over the rows the others do not cover
    uint64_t size;
    void *dest;       // the value at point i goes to dest[i·stride]
    uint64_t stride;
    // blocks·nTemp·EXPRESSION_ROWS elements for the temporaries, or null to keep them in shared
    // memory, nTemp·EXPRESSION_ROWS elements per block.
    void *temporaries;
    OperandTables tables;
};

} // namespace PilFflonk

extern "C" {

// The code of `launch` on every point of its domain: each op into its temporary, as
// Expressions::calculate computes it, and the value of the last op into dest.
void pilfflonk_gpu_calculate_expression(const PilFflonk::ExpressionLaunch *launch);

// Zi of a firstRow or lastRow boundary on the `size` points x_i = shift·ω^i of a part of the coset
// (ExpressionsDomain::cosetPart): out[i] = zh[i & zhMask]·(x_i − root)^−1, with ω^i =
// bases[i >> 8]·powers[i & 255] (gpu_plonk_precompute_omega_tables_async's tables of ω, blocks of
// 256). shift and root are host pointers to an element.
void pilfflonk_gpu_one_row_zerofier(void *out, uint64_t size, const void *shift, const void *bases,
                                    const void *powers, const void *root, const void *zh, uint64_t zhMask);

// Zi of an everyFrame on those points: out[i] = Π_{j<nRoots} (x_i − roots[j]), in that order.
void pilfflonk_gpu_frame_zerofier(void *out, uint64_t size, const void *shift, const void *bases, const void *powers,
                                  const void *roots, uint64_t nRoots);

} // extern "C"

#endif
