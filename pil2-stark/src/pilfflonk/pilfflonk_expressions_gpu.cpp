// The prover mode of the interpreter on the device (pilfflonk_expressions_gpu.hpp): compiled into
// the GPU library only (the Makefile's %_gpu.cpp rule), with g++. It calls the kernels of
// pilfflonk_expressions.cu and the PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu) through
// their C linkage.
#include "pilfflonk_expressions_gpu.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "pilfflonk_kernels.hpp"

// The PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu), declared as plonk_prover_gpu.c.cuh
// declares them.
extern "C" void gpu_plonk_memcpy_h2d(void *dst, const void *src, size_t bytes);
extern "C" void gpu_plonk_precompute_omega_tables_async(void *dBases, void *dTid, const void *omega4xPtr,
                                                        uint32_t blockSize, uint32_t numBlocks, void *stream);

namespace PilFflonk {

static_assert(uint32_t(EXPRESSION_ADD) == uint32_t(OP_ADD) && uint32_t(EXPRESSION_SUB) == uint32_t(OP_SUB) &&
                  uint32_t(EXPRESSION_MUL) == uint32_t(OP_MUL) &&
                  uint32_t(EXPRESSION_SUB_SWAP) == uint32_t(OP_SUB_SWAP),
              "the kernel's ops are the bytecode's");
static_assert(ARGS_PER_OP == 8, "the kernel reads an op as two uint4");
static_assert(sizeof(FrElement) == OPERAND_BYTES, "an operand is one element");

namespace {

// The blocks of gpu_plonk_precompute_omega_tables_async's tables, ω^(256·b) and ω^t for t < 256.
constexpr uint64_t TABLE_BLOCK = 256;
// The scalar types, publics() … challenges().
constexpr uint32_t N_SCALAR_TYPES = 6;
// The blocks of a launch whose temporaries are in device memory, per multiprocessor: as many as the
// device runs at once, about.
constexpr uint32_t TEMPORARY_BLOCKS_PER_SM = 4;
// The most blocks of a launch whose temporaries are in shared memory: they stride over the rest.
constexpr uint64_t MAX_BLOCKS = uint64_t(1) << 20;

uint64_t aligned8(uint64_t bytes) { return (bytes + 7) & ~uint64_t(7); }

// Where cosetPart keeps its tables and its Zi in its workspace, in elements: the tables of ω_S,
// then zh, zhInv and the roots of each everyFrame one after another, as they go up in one copy, and
// then the S values of each boundary's Zi but an everyRow's, which is zhInv.
struct PartLayout {
    uint64_t bases = 0;
    uint64_t powers = 0;
    uint64_t zh = 0;
    uint64_t zhInv = 0;
    std::vector<uint64_t> roots;  // by boundary
    std::vector<uint64_t> values; // by boundary
    uint64_t elements = 0;
};

PartLayout partLayout(uint64_t nBits, uint64_t partBits, const std::vector<Boundary> &boundaries) {
    const uint64_t S = uint64_t(1) << partBits;
    const uint64_t e = S >> nBits;
    PartLayout l;
    auto take = [&](uint64_t n) {
        const uint64_t start = l.elements;
        l.elements += n;
        return start;
    };
    l.bases = take(S / TABLE_BLOCK + 1);
    l.powers = take(TABLE_BLOCK);
    l.zh = take(e);
    l.zhInv = take(e);
    for (const Boundary &b : boundaries) {
        l.roots.push_back(take(b.type == BoundaryType::EveryFrame ? b.offsetMin + b.offsetMax : 0));
    }
    for (const Boundary &b : boundaries) {
        l.values.push_back(b.type == BoundaryType::EveryRow ? l.zhInv : take(S));
    }
    return l;
}

// The bytes of a block's temporaries in device memory for the code of `bin`: 0 if they fit in
// sharedBytes of shared memory.
uint64_t temporaryBytesPerBlock(const ExpressionsBin &bin, uint64_t sharedBytes) {
    const uint64_t perBlock = uint64_t(bin.maxTmp) * EXPRESSION_ROWS * OPERAND_BYTES;
    return perBlock > sharedBytes ? perBlock : 0;
}

bool isProverScalar(const OperandTypes &types, uint32_t type) {
    return type >= types.publics() && type <= types.challenges() && type != types.numbers();
}

} // namespace

// ---------------------------------------------------------------------------------------------
// ExpressionsDomainGpu
// ---------------------------------------------------------------------------------------------

ExpressionsDomainGpu ExpressionsDomainGpu::trace(uint64_t nBits) {
    const ExpressionsDomain h = ExpressionsDomain::trace(nBits);
    return ExpressionsDomainGpu(h.nBits(), h.extendBits());
}

uint64_t ExpressionsDomainGpu::cosetPartBytes(uint64_t nBits, uint64_t partBits,
                                              const std::vector<Boundary> &boundaries) {
    return partLayout(nBits, partBits, boundaries).elements * sizeof(FrElement);
}

ExpressionsDomainGpu ExpressionsDomainGpu::cosetPart(uint64_t nBits, uint64_t nBitsExt, uint64_t partBits,
                                                     uint64_t part, const std::vector<Boundary> &boundaries,
                                                     void *workspace) {
    const CosetPart points = cosetPartPoints(nBits, nBitsExt, partBits, part, boundaries);
    if (workspace == nullptr) {
        throw std::invalid_argument("ExpressionsDomainGpu::cosetPart: workspace is null");
    }
    const uint64_t S = uint64_t(1) << partBits;
    const uint64_t e = points.zh.size();
    const PartLayout l = partLayout(nBits, partBits, boundaries);
    std::vector<FrElement> tables(points.zh);
    tables.insert(tables.end(), points.zhInv.begin(), points.zhInv.end());
    for (const Boundary &b : boundaries) {
        if (b.type == BoundaryType::EveryFrame) {
            const std::vector<FrElement> roots = excludedRoots(nBits, b);
            tables.insert(tables.end(), roots.begin(), roots.end());
        }
    }
    FrElement *w = static_cast<FrElement *>(workspace);
    gpu_plonk_memcpy_h2d(w + l.zh, tables.data(), tables.size() * sizeof(FrElement));
    gpu_plonk_precompute_omega_tables_async(w + l.bases, w + l.powers, &points.root, TABLE_BLOCK,
                                            static_cast<uint32_t>(S / TABLE_BLOCK + 1), nullptr);

    ExpressionsDomainGpu domain(nBits, partBits - nBits);
    for (size_t b = 0; b < boundaries.size(); ++b) {
        const Boundary &boundary = boundaries[b];
        FrElement *zi = w + l.values[b];
        switch (boundary.type) {
        case BoundaryType::EveryRow:
            break;
        case BoundaryType::FirstRow:
        case BoundaryType::LastRow: {
            const FrElement root = oneRowRoot(nBits, boundary.type);
            pilfflonk_gpu_one_row_zerofier(zi, S, &points.shift, w + l.bases, w + l.powers, &root, w + l.zh, e - 1);
            break;
        }
        case BoundaryType::EveryFrame:
            pilfflonk_gpu_frame_zerofier(zi, S, &points.shift, w + l.bases, w + l.powers, w + l.roots[b],
                                         boundary.offsetMin + boundary.offsetMax);
            break;
        }
        domain.zerofiers_.push_back(zi);
        domain.masks_.push_back(boundary.type == BoundaryType::EveryRow ? e - 1 : S - 1);
    }
    return domain;
}

// ---------------------------------------------------------------------------------------------
// The operand tables
// ---------------------------------------------------------------------------------------------

OperandLayout operandLayout(const ExpressionsBin &bin, uint64_t nOpenings, uint64_t nBoundaries) {
    const OperandTypes types = bin.types();
    OperandLayout l;
    std::vector<uint64_t> nColumns(types.zi(), 0);
    l.nScalars.assign(N_SCALAR_TYPES, 0);
    for (const ParserArgs *code : {&bin.expressionsBinArgsExpressions, &bin.expressionsBinArgsConstraints}) {
        for (uint64_t k = 0; k + ARGS_PER_OP <= code->args.size(); k += ARGS_PER_OP) {
            for (int s = 0; s < 2; ++s) {
                const uint32_t type = code->args[k + 2 + 3 * s];
                const uint64_t count = uint64_t(code->args[k + 3 + 3 * s]) + 1;
                if (types.isColumn(type)) {
                    nColumns[type] = std::max(nColumns[type], count);
                } else if (isProverScalar(types, type)) {
                    uint64_t &n = l.nScalars[type - types.publics()];
                    n = std::max(n, count);
                }
            }
        }
    }
    l.columnStart.push_back(0);
    for (uint64_t n : nColumns) {
        l.columnStart.push_back(static_cast<uint32_t>(l.columnStart.back() + n));
    }
    l.nShifts = nOpenings;
    l.nZerofiers = nBoundaries;

    uint64_t nValues = 0;
    for (uint64_t n : l.nScalars) {
        nValues += n;
    }
    uint64_t end = 0;
    auto take = [&](uint64_t bytes) {
        const uint64_t start = end;
        end = aligned8(end + bytes);
        return start;
    };
    l.values = take(nValues * sizeof(FrElement));
    l.scalars = take(N_SCALAR_TYPES * sizeof(void *));
    l.columns = take(uint64_t(l.columnStart.back()) * sizeof(void *));
    l.shifts = take(nOpenings * sizeof(uint64_t));
    l.zerofiers = take(nBoundaries * sizeof(void *));
    l.masks = take(nBoundaries * sizeof(uint64_t));
    l.columnStarts = take(l.columnStart.size() * sizeof(uint32_t));
    l.bytes = end;
    return l;
}

void encodeOperands(const OperandLayout &layout, const OperandTypes &types, const ProverValues &values,
                    const void *numbers, const std::vector<uint64_t> &shifts,
                    const std::vector<const void *> &zerofiers, const std::vector<uint64_t> &zerofierMasks,
                    uint8_t *out, const void *base) {
    std::memset(out, 0, layout.bytes);
    auto put = [&](uint64_t offset, const void *src, uint64_t bytes) {
        if (bytes > 0) {
            std::memcpy(out + offset, src, bytes);
        }
    };
    auto putPointer = [&](uint64_t offset, const void *pointer) { put(offset, &pointer, sizeof(pointer)); };

    // The scalars' values, and where each type's are; the numbers are the code's.
    uint64_t value = layout.values;
    for (uint32_t k = 0; k < N_SCALAR_TYPES; ++k) {
        const uint32_t type = types.publics() + k;
        if (type == types.numbers()) {
            putPointer(layout.scalars + k * sizeof(void *), numbers);
            continue;
        }
        putPointer(layout.scalars + k * sizeof(void *), static_cast<const uint8_t *>(base) + value);
        const std::vector<FrElement> &given = *proverScalars(types, type, values);
        put(value, given.data(), std::min<uint64_t>(given.size(), layout.nScalars[k]) * sizeof(FrElement));
        value += layout.nScalars[k] * sizeof(FrElement);
    }
    for (uint64_t type = 0; type + 1 < layout.columnStart.size(); ++type) {
        for (uint64_t j = layout.columnStart[type]; j < layout.columnStart[type + 1]; ++j) {
            const uint64_t arg1 = j - layout.columnStart[type];
            const bool given = type < values.columns.size() && arg1 < values.columns[type].size();
            putPointer(layout.columns + j * sizeof(void *), given ? values.columns[type][arg1] : nullptr);
        }
    }
    put(layout.shifts, shifts.data(), std::min<uint64_t>(shifts.size(), layout.nShifts) * sizeof(uint64_t));
    const uint64_t nZerofiers = std::min<uint64_t>(zerofiers.size(), layout.nZerofiers);
    put(layout.zerofiers, zerofiers.data(), nZerofiers * sizeof(void *));
    put(layout.masks, zerofierMasks.data(), nZerofiers * sizeof(uint64_t));
    put(layout.columnStarts, layout.columnStart.data(), layout.columnStart.size() * sizeof(uint32_t));
}

OperandTables operandTables(const OperandLayout &layout, const OperandTypes &types, const void *base, uint64_t size) {
    const uint8_t *at = static_cast<const uint8_t *>(base);
    OperandTables t;
    t.columns = reinterpret_cast<const void *const *>(at + layout.columns);
    t.columnStart = reinterpret_cast<const uint32_t *>(at + layout.columnStarts);
    t.shifts = reinterpret_cast<const uint64_t *>(at + layout.shifts);
    t.zerofiers = reinterpret_cast<const void *const *>(at + layout.zerofiers);
    t.zerofierMasks = reinterpret_cast<const uint64_t *>(at + layout.masks);
    t.scalars = reinterpret_cast<const void *const *>(at + layout.scalars);
    t.mask = size - 1;
    t.ziType = types.zi();
    t.scalarsType = types.publics();
    return t;
}

// ---------------------------------------------------------------------------------------------
// ExpressionsGpu
// ---------------------------------------------------------------------------------------------

ExpressionsGpu::ExpressionsGpu(const ExpressionsBin &bin, const PilfflonkInfo &info, DeviceCode expressions,
                               DeviceCode constraints, uint64_t _sharedBytes)
    : host(bin, info), types(bin.types()), expressionsCode(expressions), constraintsCode(constraints),
      layout(operandLayout(bin, info.openingPoints.size(), info.boundaries.size())), sharedBytes(_sharedBytes) {
    if (sharedBytes > MAX_SHARED_BYTES) {
        throw std::invalid_argument("ExpressionsGpu: " + std::to_string(sharedBytes) +
                                    " bytes of shared memory per block, and a block may take " +
                                    std::to_string(MAX_SHARED_BYTES));
    }
    tables = DeviceBuffer(layout.bytes);
    const uint64_t perBlock = temporaryBytesPerBlock(bin, sharedBytes);
    if (perBlock > 0) {
        temporaryBlocks = pilfflonk_gpu_multiprocessors() * TEMPORARY_BLOCKS_PER_SM;
        temporaries = DeviceBuffer(temporaryBlocks * perBlock);
    }
}

uint64_t ExpressionsGpu::deviceBytesOf(const ExpressionsBin &bin, const PilfflonkInfo &info, uint32_t multiprocessors,
                                       uint64_t sharedBytes) {
    return operandLayout(bin, info.openingPoints.size(), info.boundaries.size()).bytes +
           uint64_t(multiprocessors) * TEMPORARY_BLOCKS_PER_SM * temporaryBytesPerBlock(bin, sharedBytes);
}

void ExpressionsGpu::calculateExpression(uint64_t expId, const ExpressionsDomainGpu &domain, const ProverValues &values,
                                         FrElement *dest, uint64_t stride) const {
    calculate(host.checkedExpression(expId, domain.nZerofiers(), values, dest), expressionsCode, domain, values, dest,
              stride, "calculateExpression");
}

void ExpressionsGpu::calculateConstraint(uint64_t index, const ExpressionsDomainGpu &domain, const ProverValues &values,
                                         FrElement *dest, uint64_t stride) const {
    if (constraintsCode.args == nullptr) {
        throw std::logic_error("ExpressionsGpu::calculateConstraint: the constraints' code is not on the device");
    }
    calculate(host.checkedConstraint(index, domain.nZerofiers(), values, dest), constraintsCode, domain, values, dest,
              stride, "calculateConstraint");
}

void ExpressionsGpu::calculate(const ParserParams &params, const DeviceCode &code, const ExpressionsDomainGpu &domain,
                               const ProverValues &values, FrElement *dest, uint64_t stride,
                               const char *function) const {
    if (stride == 0) {
        throw std::invalid_argument(std::string("ExpressionsGpu::") + function + ": a stride of 0");
    }
    std::vector<uint8_t> bytes(layout.bytes);
    encodeOperands(layout, types, values, code.numbers, host.shifts(domain.size(), domain.extendBits()),
                   domain.zerofiers(), domain.zerofierMasks(), bytes.data(), tables.data());
    // Ordered after the evaluations before, which read the tables.
    gpu_plonk_memcpy_h2d(tables.data(), bytes.data(), bytes.size());

    const bool inShared = uint64_t(params.nTemp) * EXPRESSION_ROWS * OPERAND_BYTES <= sharedBytes;
    const uint64_t chunks = (domain.size() + EXPRESSION_ROWS - 1) / EXPRESSION_ROWS;
    ExpressionLaunch launch{};
    launch.args = code.args + params.argsOffset;
    launch.nOps = params.nOps;
    launch.nTemp = params.nTemp;
    launch.tmpType = types.tmp();
    launch.blocks = static_cast<uint32_t>(std::min<uint64_t>(chunks, inShared ? MAX_BLOCKS : temporaryBlocks));
    launch.size = domain.size();
    launch.dest = dest;
    launch.stride = stride;
    launch.temporaries = inShared ? nullptr : temporaries.data();
    launch.tables = operandTables(layout, types, tables.data(), domain.size());
    pilfflonk_gpu_calculate_expression(&launch);
}

} // namespace PilFflonk
