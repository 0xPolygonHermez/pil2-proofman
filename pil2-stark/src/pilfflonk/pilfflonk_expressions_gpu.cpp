// The prover mode of the interpreter on the device (pilfflonk_expressions_gpu.hpp): compiled into
// the GPU library only (the Makefile's %_gpu.cpp rule), with g++. It calls the kernels of
// pilfflonk_expressions.cu and the PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu) through
// their C linkage.
#include "pilfflonk_expressions_gpu.hpp"

#include <dlfcn.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "pilfflonk_kernels.hpp"
#include "zklog.hpp"

namespace PilFflonk {

static_assert(uint32_t(EXPRESSION_ADD) == uint32_t(OP_ADD) && uint32_t(EXPRESSION_SUB) == uint32_t(OP_SUB) &&
                  uint32_t(EXPRESSION_MUL) == uint32_t(OP_MUL) &&
                  uint32_t(EXPRESSION_SUB_SWAP) == uint32_t(OP_SUB_SWAP),
              "the kernel's ops are the bytecode's");
static_assert(ARGS_PER_OP == 8, "the kernel reads an op as two uint4");
static_assert(sizeof(FrElement) == OPERAND_BYTES, "an operand is one element");

namespace {

// The scalar types, publics() … challenges().
constexpr uint32_t N_SCALAR_TYPES = 6;
// The blocks of a launch whose temporaries are in device memory, per multiprocessor: as many as the
// device runs at once, about.
constexpr uint32_t TEMPORARY_BLOCKS_PER_SM = 4;
// The most blocks of a launch whose temporaries are in shared memory: they stride over the rest.
constexpr uint64_t MAX_BLOCKS = uint64_t(1) << 20;

uint64_t aligned8(uint64_t bytes) { return (bytes + 7) & ~uint64_t(7); }

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

uint64_t ExpressionsDomainGpu::cosetPartBytes(uint64_t nBits, uint64_t partBits) {
    // 1/Z_H on the part's points: e = S/N values, repeated every e points
    return ((uint64_t(1) << partBits) >> nBits) * sizeof(FrElement);
}

ExpressionsDomainGpu ExpressionsDomainGpu::cosetPart(uint64_t nBits, uint64_t nBitsExt, uint64_t partBits,
                                                     uint64_t part, void *workspace) {
    const CosetPart points = cosetPartPoints(nBits, nBitsExt, partBits, part);
    if (workspace == nullptr) {
        throw std::invalid_argument("ExpressionsDomainGpu::cosetPart: workspace is null");
    }
    gpu_plonk_memcpy_h2d(workspace, points.zhInv.data(), points.zhInv.size() * sizeof(FrElement));
    ExpressionsDomainGpu domain(nBits, partBits - nBits);
    domain.zhInv_ = workspace;
    domain.zhMask_ = points.zhInv.size() - 1;
    return domain;
}

// ---------------------------------------------------------------------------------------------
// The operand tables
// ---------------------------------------------------------------------------------------------

OperandLayout operandLayout(const ExpressionsBin &bin, uint64_t nOpenings) {
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
    l.columnStarts = take(l.columnStart.size() * sizeof(uint32_t));
    l.bytes = end;
    return l;
}

void encodeOperands(const OperandLayout &layout, const OperandTypes &types, const ProverValues &values,
                    const void *numbers, const std::vector<uint64_t> &shifts, uint8_t *out, const void *base) {
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
    put(layout.columnStarts, layout.columnStart.data(), layout.columnStart.size() * sizeof(uint32_t));
}

OperandTables operandTables(const OperandLayout &layout, const OperandTypes &types, const void *base, uint64_t size,
                            const void *zhInv, uint64_t zhMask) {
    const uint8_t *at = static_cast<const uint8_t *>(base);
    OperandTables t;
    t.columns = reinterpret_cast<const void *const *>(at + layout.columns);
    t.columnStart = reinterpret_cast<const uint32_t *>(at + layout.columnStarts);
    t.shifts = reinterpret_cast<const uint64_t *>(at + layout.shifts);
    t.zhInv = zhInv;
    t.zhMask = zhMask;
    t.scalars = reinterpret_cast<const void *const *>(at + layout.scalars);
    t.mask = size - 1;
    t.ziType = types.zi();
    t.scalarsType = types.publics();
    return t;
}

// ---------------------------------------------------------------------------------------------
// GeneratedExpressions
// ---------------------------------------------------------------------------------------------

namespace {
// setup/exps-codegen/src/bn128.rs's PFEXPS_ABI.
constexpr unsigned GENERATED_ABI = 2;
} // namespace

GeneratedExpressions::GeneratedExpressions(const std::string &binPath) {
    const std::string ext = ".bin";
    const char *env = std::getenv("PILFFLONK_EXPS");
    if ((env != nullptr && std::string(env) == "0") || binPath.size() < ext.size() ||
        binPath.compare(binPath.size() - ext.size(), ext.size(), ext) != 0) {
        return;
    }
    path = binPath.substr(0, binPath.size() - ext.size()) + ".exps.so";
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) {
        return;
    }
    lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        throw std::runtime_error("GeneratedExpressions: cannot load " + path + ": " + dlerror());
    }
    auto abi = reinterpret_cast<unsigned (*)()>(dlsym(lib, "pfexps_abi"));
    auto scratchFn = reinterpret_cast<unsigned long long (*)()>(dlsym(lib, "pfexps_scratch_bytes"));
    coversFn = reinterpret_cast<int (*)(unsigned)>(dlsym(lib, "pfexps_covers"));
    launchFn = reinterpret_cast<int (*)(unsigned, const ExpressionLaunch *, void *)>(dlsym(lib, "pfexps_launch"));
    if (abi == nullptr || scratchFn == nullptr || coversFn == nullptr || launchFn == nullptr ||
        abi() != GENERATED_ABI) {
        dlclose(lib);
        throw std::runtime_error("GeneratedExpressions: " + path +
                                 " is not a pilfflonk expressions library of this ABI: regenerate it (gen-exps)");
    }
    scratch = scratchFn();
}

GeneratedExpressions::~GeneratedExpressions() {
    if (lib != nullptr) {
        dlclose(lib);
    }
}

bool GeneratedExpressions::covers(uint64_t expId) const {
    return lib != nullptr && expId <= UINT32_MAX && coversFn(static_cast<unsigned>(expId)) == 1;
}

void GeneratedExpressions::launch(uint64_t expId, const ExpressionLaunch &l, void *scratchMemory) const {
    const int err = launchFn(static_cast<unsigned>(expId), &l, scratchMemory);
    if (err != 0) {
        throw std::runtime_error("GeneratedExpressions: " + path + " failed to launch expression " +
                                 std::to_string(expId) + " (error " + std::to_string(err) + ")");
    }
}

// ---------------------------------------------------------------------------------------------
// ExpressionsGpu
// ---------------------------------------------------------------------------------------------

ExpressionsGpu::ExpressionsGpu(const ExpressionsBin &bin, const PilfflonkInfo &info, DeviceCode expressions,
                               uint64_t _sharedBytes)
    : host(bin, info), types(bin.types()), expressionsCode(expressions),
      layout(operandLayout(bin, info.openingPoints.size())), sharedBytes(_sharedBytes) {
    if (sharedBytes > MAX_SHARED_BYTES) {
        throw std::invalid_argument("ExpressionsGpu: " + std::to_string(sharedBytes) +
                                    " bytes of shared memory per block, and a block may take " +
                                    std::to_string(MAX_SHARED_BYTES));
    }
    tables = DeviceBuffer(layout.bytes);
    generated = std::make_unique<GeneratedExpressions>(bin.path);
    if (generated->loaded()) {
        zklog.info("pilfflonk: generated expression kernels from " + generated->file());
    }
    if (generated->scratchBytes() > 0) {
        generatedScratch = DeviceBuffer(generated->scratchBytes());
    }
    const uint64_t perBlock = temporaryBytesPerBlock(bin, sharedBytes);
    if (perBlock > 0) {
        temporaryBlocks = pilfflonk_gpu_multiprocessors() * TEMPORARY_BLOCKS_PER_SM;
        temporaries = DeviceBuffer(temporaryBlocks * perBlock);
    }
}

uint64_t ExpressionsGpu::deviceBytesOf(const ExpressionsBin &bin, const PilfflonkInfo &info, uint32_t multiprocessors,
                                       uint64_t sharedBytes) {
    return operandLayout(bin, info.openingPoints.size()).bytes +
           uint64_t(multiprocessors) * TEMPORARY_BLOCKS_PER_SM * temporaryBytesPerBlock(bin, sharedBytes) +
           GeneratedExpressions(bin.path).scratchBytes();
}

void ExpressionsGpu::calculateExpression(uint64_t expId, const ExpressionsDomainGpu &domain, const ProverValues &values,
                                         FrElement *dest, uint64_t stride) const {
    const ParserParams &params = host.checkedExpression(expId, domain.hasZi(), values, dest);
    if (stride == 0) {
        throw std::invalid_argument("ExpressionsGpu::calculateExpression: a stride of 0");
    }
    const DeviceCode &code = expressionsCode;
    std::vector<uint8_t> bytes(layout.bytes);
    encodeOperands(layout, types, values, code.numbers, host.shifts(domain.size(), domain.extendBits()), bytes.data(),
                   tables.data());
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
    launch.tables = operandTables(layout, types, tables.data(), domain.size(), domain.zhInv(), domain.zhMask());
    if (generated->covers(expId)) {
        generated->launch(expId, launch, generatedScratch.data());
        return;
    }
    pilfflonk_gpu_calculate_expression(&launch);
}

} // namespace PilFflonk
