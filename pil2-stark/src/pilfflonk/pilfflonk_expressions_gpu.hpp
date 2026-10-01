#ifndef PILFFLONK_EXPRESSIONS_GPU_HPP
#define PILFFLONK_EXPRESSIONS_GPU_HPP

// The prover mode of the interpreter on the device (pilfflonk/docs/performance.md#gpu): the code of
// an AIR's <air>.bin on columns in device memory, into device memory, by the kernels of
// pilfflonk_expressions.cu; and the domains it runs on, H and the parts of the extended coset, with
// their Zi computed on the device. Only a library built with the GPU (__USE_CUDA__) has it, in
// pilfflonk_expressions_gpu.cpp.
//
// Every value is the CPU's bit for bit (Expressions, ExpressionsDomain): the same field operations
// on the same elements, in sppark's arithmetic, which keeps every element fully reduced as ffiasm
// does; the host math they share (cosetPartPoints, excludedRoots, oneRowRoot,
// Expressions::shifts); and the CPU's own checks, with its errors, before anything is computed.
// Everything runs on the legacy default stream, after what the caller has queued there.

#include <cstdint>
#include <vector>

#include "pilfflonk_expressions.hpp"
#include "pilfflonk_expressions_kernels.hpp"
#include "pilfflonk_key_gpu.hpp"

namespace PilFflonk {

// A domain of the prover mode on the device (ExpressionsDomain): H, or a part of the extended coset
// and Zi of each boundary on its points, in device memory.
class ExpressionsDomainGpu {
public:
    // H, as ExpressionsDomain::trace, which throws as it does.
    static ExpressionsDomainGpu trace(uint64_t nBits);

    // ExpressionsDomain::cosetPart on the device: its arguments checked, and Z_H on its points
    // computed, by cosetPartPoints, and then Zi of each boundary on the device into `workspace`,
    // cosetPartBytes() bytes of device memory, which must outlive the domain and the evaluations on
    // it. Zi of an everyRow, 1/Z_H, repeats every e = S/N points, and has its e values only. Throws
    // as cosetPart, and std::invalid_argument if workspace is null. Returns before the kernels
    // finish: the default stream runs them before the evaluations on the domain.
    static ExpressionsDomainGpu cosetPart(uint64_t nBits, uint64_t nBitsExt, uint64_t partBits, uint64_t part,
                                          const std::vector<Boundary> &boundaries, void *workspace);

    // The workspace of cosetPart, in bytes, for boundaries it accepts.
    static uint64_t cosetPartBytes(uint64_t nBits, uint64_t partBits, const std::vector<Boundary> &boundaries);

    uint64_t nBits() const { return nBits_; }
    uint64_t size() const { return uint64_t(1) << (nBits_ + extendBits_); }
    uint64_t extendBits() const { return extendBits_; }
    uint64_t nZerofiers() const { return zerofiers_.size(); }
    // Zi of boundary b at point i is zerofiers()[b][i & zerofierMasks()[b]], on the device.
    const std::vector<const void *> &zerofiers() const { return zerofiers_; }
    const std::vector<uint64_t> &zerofierMasks() const { return masks_; }

private:
    ExpressionsDomainGpu(uint64_t nBits, uint64_t extendBits) : nBits_(nBits), extendBits_(extendBits) {}

    uint64_t nBits_;
    uint64_t extendBits_;
    std::vector<const void *> zerofiers_;
    std::vector<uint64_t> masks_;
};

// The device's copy of the code of a section of <air>.bin (ParserArgs): its args, 16-byte aligned,
// and its numbers.
struct DeviceCode {
    const uint32_t *args = nullptr;
    const FrElement *numbers = nullptr;
};

// Where the operand tables of an evaluation (OperandTables) go in the bytes encodeOperands writes:
// room for every column and scalar the code of a bytecode reads, in both its sections, and for
// every opening point and boundary of its AIR. Offsets in bytes.
struct OperandLayout {
    std::vector<uint32_t> columnStart; // by column type, 0 … zi() − 1, then their entries in all
    std::vector<uint64_t> nScalars;    // by scalar type − publics(): 1 + the largest arg1 read; 0 for the numbers
    uint64_t nShifts = 0;
    uint64_t nZerofiers = 0;
    uint64_t values = 0;  // the scalars' values, nScalars[k] of each type k one after another
    uint64_t scalars = 0; // OperandTables::scalars
    uint64_t columns = 0;
    uint64_t shifts = 0;
    uint64_t zerofiers = 0;
    uint64_t masks = 0;
    uint64_t columnStarts = 0;
    uint64_t bytes = 0;
};

OperandLayout operandLayout(const ExpressionsBin &bin, uint64_t nOpenings, uint64_t nBoundaries);

// The tables of an evaluation, as `layout` places them in layout.bytes bytes at `out`, to be read at
// `base` (where they will be copied, or out itself): the columns of `values` (pointers, kept as they
// are, null past what it has), its scalars (their values, which the tables point to, 0 past what it
// has), the code's numbers at `numbers`, the shifts of the opening points (Expressions::shifts) and
// the domain's Zi (ExpressionsDomainGpu::zerofiers and zerofierMasks). The code must have been
// checked against the values (Expressions::checkedExpression): it reads nothing they do not have.
void encodeOperands(const OperandLayout &layout, const OperandTypes &types, const ProverValues &values,
                    const void *numbers, const std::vector<uint64_t> &shifts,
                    const std::vector<const void *> &zerofiers, const std::vector<uint64_t> &zerofierMasks,
                    uint8_t *out, const void *base);

// Those tables at `base`, on a domain of `size` points.
OperandTables operandTables(const OperandLayout &layout, const OperandTypes &types, const void *base, uint64_t size);

// Expressions on the device: the code of an AIR's .bin, evaluated by the kernel of
// pilfflonk_expressions.cu with its operand tables, which it writes before each evaluation. The
// temporaries of a code block go in shared memory if they fit (EXPRESSION_ROWS·nTemp elements per
// block within sharedBytes), and otherwise in device memory it holds for them, which fewer blocks
// share: the values are the same either way. Not safe from several threads at once, as its tables
// are one; the holder of a GpuKey's Lease uses it, or a test.
class ExpressionsGpu {
public:
    // The most shared memory a block may take without asking for more: 48 KiB.
    static constexpr uint64_t MAX_SHARED_BYTES = uint64_t(48) << 10;

    // The code of `bin` for the AIR `info`, checked as Expressions checks it (FormatError), with the
    // code of its expressions on the device at `expressions` (a GpuAirKey's args() and numbers()),
    // and that of its constraints at `constraints` (none: calculateConstraint refuses). It keeps a
    // reference to `bin`, which must outlive it. Throws std::invalid_argument if sharedBytes exceeds
    // MAX_SHARED_BYTES.
    ExpressionsGpu(const ExpressionsBin &bin, const PilfflonkInfo &info, DeviceCode expressions,
                   DeviceCode constraints = DeviceCode(), uint64_t sharedBytes = MAX_SHARED_BYTES);

    // Expressions::calculateExpression on the device: expression expId on every point of `domain`,
    // the value at point i into dest[i·stride], device memory. values.columns are device pointers,
    // each to the domain.size() values of its column in the domain's order, and none overlaps dest;
    // its scalars are on the host, and are copied. It checks and throws what calculateExpression
    // does (Expressions::checkedExpression), with its messages, and std::invalid_argument for a
    // stride of 0. Returns before the kernel finishes.
    void calculateExpression(uint64_t expId, const ExpressionsDomainGpu &domain, const ProverValues &values,
                             FrElement *dest, uint64_t stride = 1) const;

    // The same for constraint `index` of section 2 (Expressions::calculateConstraint). Throws
    // std::logic_error, first, if it was given no code of the constraints.
    void calculateConstraint(uint64_t index, const ExpressionsDomainGpu &domain, const ProverValues &values,
                             FrElement *dest, uint64_t stride = 1) const;

    // The device memory it holds, in bytes: its tables, and the temporaries that do not fit in
    // shared memory.
    uint64_t deviceBytes() const { return tables.size() + temporaries.size(); }

private:
    void calculate(const ParserParams &params, const DeviceCode &code, const ExpressionsDomainGpu &domain,
                   const ProverValues &values, FrElement *dest, uint64_t stride, const char *function) const;

    Expressions host;
    OperandTypes types;
    DeviceCode expressionsCode;
    DeviceCode constraintsCode;
    OperandLayout layout;
    uint64_t sharedBytes;
    uint32_t temporaryBlocks = 0; // the blocks of a launch whose temporaries are in `temporaries`
    DeviceBuffer tables;
    DeviceBuffer temporaries;
};

} // namespace PilFflonk

#endif
