// Tests of the interpreter's prover mode on the device (pilfflonk_expressions_gpu.hpp), in a library
// built with the GPU: every expression and constraint of every bytecode under test, on H and on
// every part of the extended coset of 2^partBits points for partBits = nBits, nBits + 1 and
// nBitsExt, against Expressions::calculateExpression and calculateConstraint byte for byte, with
// their temporaries in shared memory and in device memory, into a strided dest on the parts (as Q
// goes into the coset's order); the errors too, which are the CPU's word for word; and the Zi of
// every kind of boundary (ExpressionsDomainGpu) against ExpressionsDomain::cosetPart. The columns
// and scalars are random, from a fixed seed, with 0, 1 and r − 1 among them: the code does not
// need a witness that satisfies it to be computed the same.
//
// Where there is no GPU (gpuUnderTest), the host's part still runs: the tables of the operands
// (encodeOperands), read on the host through operandAddress as the kernel reads them, give the
// CPU's values on the same domains, for the same bytecodes.
//
// The bytecodes are setup-pilfflonk's fixtures of setup/pilfflonk/tests/fixtures/bytecode/, and
// those of every <air>.bin with its <air>.pilfflonkinfo.json under the directories of
// PILFFLONK_GPU_EXPRESSION_KEYS (colon-separated): proving keys, or the two files of each. A key of
// more than 2^TEST_NBITS rows is tested on domains of 2^TEST_NBITS rows, with its extension, which
// runs every op of its code; Q of the one key PILFFLONK_GPU_TIMING_KEY names (a provingKey/) runs
// at its size, on its parts of N and 2N points, timed against the CPU.
#include "pilfflonk_test.hpp"

#ifdef __USE_CUDA__

#include <limits.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pilfflonk_expressions.hpp"
#include "pilfflonk_expressions_bin.hpp"
#include "pilfflonk_expressions_gpu.hpp"
#include "pilfflonk_info.hpp"
#include "pilfflonk_proving_key.hpp"

// The PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu).
extern "C" void gpu_plonk_memcpy_h2d(void *dst, const void *src, size_t bytes);
extern "C" void gpu_plonk_memcpy_d2h(void *dst, const void *src, size_t bytes);
extern "C" void gpu_plonk_cuda_device_sync();

#endif

namespace PilFflonkTest {

#ifdef __USE_CUDA__

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using PilFflonk::Boundary;
using PilFflonk::BoundaryType;
using PilFflonk::DeviceBuffer;
using PilFflonk::DeviceCode;
using PilFflonk::Expressions;
using PilFflonk::ExpressionsBin;
using PilFflonk::ExpressionsDomain;
using PilFflonk::ExpressionsDomainGpu;
using PilFflonk::ExpressionsGpu;
using PilFflonk::FrElement;
using PilFflonk::OperandLayout;
using PilFflonk::OperandTables;
using PilFflonk::OperandTypes;
using PilFflonk::ParserArgs;
using PilFflonk::ParserParams;
using PilFflonk::PilfflonkInfo;
using PilFflonk::ProverValues;
using Engine = AltBn128::Engine;
using Column = std::vector<FrElement>;

Engine &E = Engine::engine;

// The rows of the domains a larger key is tested on.
constexpr uint64_t TEST_NBITS = 10;

std::string repoPath(const std::string &relative) {
    if (const char *root = std::getenv("PILFFLONK_REPO_ROOT")) {
        return std::string(root) + "/" + relative;
    }
    char exe[PATH_MAX];
    const ssize_t length = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    assert(length > 0);
    exe[length] = '\0';
    const std::string dir(exe);
    return dir.substr(0, dir.rfind('/')) + "/../../" + relative;
}

std::string fixture(const std::string &name) { return repoPath("setup/pilfflonk/tests/fixtures/bytecode/" + name); }

// Elements from a fixed seed, below 2^253 < r as Montgomery limbs; a column has 0, 1, r − 1 and a
// repeated one among them, rotated to a random row.
class Random {
public:
    explicit Random(uint64_t seed) : generator(seed) {}

    FrElement element() {
        FrElement e;
        for (uint64_t &limb : e.v) {
            limb = generator();
        }
        e.v[3] >>= 3;
        return e;
    }

    Column column(uint64_t n) {
        const FrElement same = element();
        const FrElement edges[] = {E.fr.zero(), E.fr.one(), E.fr.negOne(), same, same};
        Column c(n);
        for (uint64_t i = 0; i < n; ++i) {
            c[i] = i < 5 ? edges[i] : element();
        }
        std::rotate(c.begin(), c.begin() + generator() % n, c.end());
        return c;
    }

private:
    std::mt19937_64 generator;
};

bool same(const FrElement *a, const FrElement *b, uint64_t n) { return std::memcmp(a, b, n * sizeof(FrElement)) == 0; }

DeviceBuffer upload(const void *data, uint64_t bytes) {
    DeviceBuffer device(bytes);
    gpu_plonk_memcpy_h2d(device.data(), data, bytes);
    return device;
}

Column download(const void *device, uint64_t n) {
    Column host(n);
    gpu_plonk_memcpy_d2h(host.data(), device, n * sizeof(FrElement));
    return host;
}

// What `call` throws, as "<kind>: <message>", or "" if it throws nothing.
template <typename Call>
std::string outcome(Call call) {
    try {
        call();
    } catch (const std::invalid_argument &e) {
        return std::string("invalid_argument: ") + e.what();
    } catch (const std::logic_error &e) {
        return std::string("logic_error: ") + e.what();
    } catch (const std::runtime_error &e) {
        return std::string("runtime_error: ") + e.what();
    }
    return "";
}

// A bytecode under test, and its AIR's description.
struct Air {
    std::string name;
    PilfflonkInfo info;
    ExpressionsBin bin;
    uint64_t nBitsExt; // the key's
};

Air loadAir(const std::string &binPath, const std::string &infoPath) {
    Air air{binPath, PilfflonkInfo::load(infoPath), ExpressionsBin::load(binPath), 0};
    air.nBitsExt = PilFflonk::airDegrees(air.info, air.name).nBitsExt;
    return air;
}

// Sample.bin, sample() of setup/pilfflonk/tests/bytecode.rs, with the AIR of Sample.expected.json:
// two stages and a boundary of every kind.
Air sampleAir() {
    const std::string path = fixture("Sample.bin");
    std::ifstream file(fixture("Sample.expected.json"));
    assert(file);
    const json j = json::parse(file);
    Air air{path, PilfflonkInfo{}, ExpressionsBin::load(path), j["nBitsExt"].get<uint64_t>()};
    air.info.nBits = j["nBits"];
    air.info.nStages = 2;
    air.info.openingPoints = j["openingPoints"].get<std::vector<int64_t>>();
    for (const json &b : j["boundaries"]) {
        const std::string name = b["name"];
        if (name == "everyRow") air.info.boundaries.push_back({BoundaryType::EveryRow, 0, 0});
        else if (name == "firstRow") air.info.boundaries.push_back({BoundaryType::FirstRow, 0, 0});
        else if (name == "lastRow") air.info.boundaries.push_back({BoundaryType::LastRow, 0, 0});
        else air.info.boundaries.push_back({BoundaryType::EveryFrame, b["offsetMin"], b["offsetMax"]});
    }
    return air;
}

// A bytecode and its AIR's pilfflonkinfo.
using BytecodeFiles = std::pair<std::string, std::string>;

// Every <air>.bin with its <air>.pilfflonkinfo.json under `dir`, in order. Asserts there is one.
std::vector<BytecodeFiles> bytecodesIn(const std::string &dir) {
    const std::string suffix = ".pilfflonkinfo.json";
    std::vector<BytecodeFiles> found;
    for (const auto &entry : fs::recursive_directory_iterator(dir)) {
        const std::string path = entry.path().string();
        if (path.size() > suffix.size() && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
            const std::string bin = path.substr(0, path.size() - suffix.size()) + ".bin";
            if (fs::exists(bin)) {
                found.emplace_back(bin, path);
            }
        }
    }
    assert(!found.empty() && "a directory under test has no bytecode");
    std::sort(found.begin(), found.end());
    return found;
}

// Those under the directories of PILFFLONK_GPU_EXPRESSION_KEYS, in order.
std::vector<BytecodeFiles> keysUnderTest() {
    std::vector<BytecodeFiles> keys;
    const char *list = std::getenv("PILFFLONK_GPU_EXPRESSION_KEYS");
    std::string dirs = list != nullptr ? list : "";
    while (!dirs.empty()) {
        const size_t colon = dirs.find(':');
        const std::vector<BytecodeFiles> found = bytecodesIn(dirs.substr(0, colon));
        keys.insert(keys.end(), found.begin(), found.end());
        dirs = colon == std::string::npos ? "" : dirs.substr(colon + 1);
    }
    return keys;
}

// The rows of the domains `air` is tested on: its own, or 2^TEST_NBITS if it has more and its
// everyFrames fit there.
uint64_t testBits(const Air &air) {
    if (air.info.nBits <= TEST_NBITS) {
        return air.info.nBits;
    }
    for (const Boundary &b : air.info.boundaries) {
        if (b.type == BoundaryType::EveryFrame && b.offsetMin + b.offsetMax > (uint64_t(1) << TEST_NBITS)) {
            return air.info.nBits;
        }
    }
    return TEST_NBITS;
}

// A code block of the bytecode: an expression's or a constraint's.
struct Code {
    bool constraint;
    uint64_t id; // expId, or the constraint's index
};

std::vector<Code> codesOf(const ExpressionsBin &bin) {
    std::vector<Code> codes;
    for (const auto &entry : bin.expressionsInfo) {
        codes.push_back({false, entry.first});
    }
    for (uint64_t c = 0; c < bin.constraintsInfoDebug.size(); ++c) {
        codes.push_back({true, c});
    }
    return codes;
}

// Random scalars of each kind, as many as `layout` has room for, into both values.
void randomScalars(const OperandLayout &layout, const OperandTypes &types, Random &random, ProverValues &host,
                   ProverValues &device) {
    auto scalars = [&](uint32_t type) {
        const uint64_t n = layout.nScalars[type - types.publics()];
        Column c = random.column(std::max<uint64_t>(n, 1));
        c.resize(n);
        return c;
    };
    host.publics = scalars(types.publics());
    host.airValues = scalars(types.airValues());
    host.proofValues = scalars(types.proofValues());
    host.airgroupValues = scalars(types.airgroupValues());
    host.challenges = scalars(types.challenges());
    device.publics = host.publics;
    device.airValues = host.airValues;
    device.proofValues = host.proofValues;
    device.airgroupValues = host.airgroupValues;
    device.challenges = host.challenges;
}

// The columns and scalars of the operands of `layout`, random, on the host and, with `gpu`, on the
// device: `size` values per column, which a domain of fewer points reads the first of.
struct Values {
    ProverValues host;
    ProverValues device;
    std::vector<Column> columns;
    std::vector<DeviceBuffer> deviceColumns;

    Values(const OperandLayout &layout, const OperandTypes &types, uint64_t size, Random &random, bool gpu) {
        host.columns.resize(layout.columnStart.size() - 1);
        device.columns.resize(host.columns.size());
        for (uint64_t j = 0; j < layout.columnStart.back(); ++j) {
            columns.push_back(random.column(size));
        }
        for (uint64_t type = 0; type < host.columns.size(); ++type) {
            for (uint64_t j = layout.columnStart[type]; j < layout.columnStart[type + 1]; ++j) {
                host.columns[type].push_back(columns[j].data());
                if (gpu) {
                    deviceColumns.push_back(upload(columns[j].data(), size * sizeof(FrElement)));
                    device.columns[type].push_back(reinterpret_cast<const FrElement *>(deviceColumns.back().data()));
                }
            }
        }
        randomScalars(layout, types, random, host, device);
    }
};

// The code of `params` on `size` points as the device reads its operands, through the tables at
// `tables`, on the host: what encodeOperands and operandAddress give the kernel.
Column throughTheTables(const ParserParams &params, const ParserArgs &args, const OperandTables &tables,
                        uint32_t tmpType, uint64_t size) {
    Column out(size);
#pragma omp parallel
    {
        Column tmp(params.nTemp);
#pragma omp for
        for (uint64_t i = 0; i < size; ++i) {
            auto source = [&](const uint32_t *s) -> const FrElement & {
                return s[0] == tmpType ? tmp[s[1]]
                                       : *static_cast<const FrElement *>(
                                             PilFflonk::operandAddress(tables, s[0], s[1], s[2], i));
            };
            FrElement t = E.fr.zero();
            for (uint64_t k = 0; k < params.nOps; ++k) {
                const uint32_t *op = &args.args[params.argsOffset + k * PilFflonk::ARGS_PER_OP];
                const FrElement &a = source(op + 2), &b = source(op + 5);
                switch (op[0]) {
                case PilFflonk::OP_ADD:
                    E.fr.add(t, a, b);
                    break;
                case PilFflonk::OP_SUB:
                    E.fr.sub(t, a, b);
                    break;
                case PilFflonk::OP_MUL:
                    E.fr.mul(t, a, b);
                    break;
                default:
                    E.fr.sub(t, b, a);
                    break;
                }
                tmp[op[1]] = t;
            }
            out[i] = t;
        }
    }
    return out;
}

// A domain the code is tested on: the CPU's, the device's, and the Zi the host reads through the
// tables (an everyRow's as its e values, as the device keeps it).
struct Domain {
    ExpressionsDomain cpu;
    std::vector<ExpressionsDomainGpu> gpu; // one, where there is a GPU
    std::vector<const void *> hostZerofiers;
    std::vector<uint64_t> hostMasks;
    std::vector<FrElement> zhInv;
};

Domain traceDomain(uint64_t nBits, bool gpu) {
    Domain d{ExpressionsDomain::trace(nBits), {}, {}, {}, {}};
    if (gpu) {
        d.gpu.push_back(ExpressionsDomainGpu::trace(nBits));
    }
    return d;
}

Domain partDomain(uint64_t nBits, uint64_t nBitsExt, uint64_t partBits, uint64_t part,
                  const std::vector<Boundary> &boundaries, std::vector<DeviceBuffer> &workspaces, bool gpu) {
    Domain d{ExpressionsDomain::cosetPart(nBits, nBitsExt, partBits, part, boundaries), {}, {}, {}, {}};
    d.zhInv = PilFflonk::cosetPartPoints(nBits, nBitsExt, partBits, part, boundaries).zhInv;
    for (uint64_t b = 0; b < boundaries.size(); ++b) {
        const bool table = boundaries[b].type == BoundaryType::EveryRow;
        d.hostZerofiers.push_back(table ? d.zhInv.data() : d.cpu.zerofier(b).data());
        d.hostMasks.push_back(table ? d.zhInv.size() - 1 : d.cpu.size() - 1);
    }
    if (gpu) {
        workspaces.emplace_back(ExpressionsDomainGpu::cosetPartBytes(nBits, partBits, boundaries));
        d.gpu.push_back(ExpressionsDomainGpu::cosetPart(nBits, nBitsExt, partBits, part, boundaries,
                                                        workspaces.back().data()));
    }
    return d;
}

// Zi of each boundary of `gpu`, downloaded and read at every point through its mask, is cpu's.
void expectTheCpusZerofiers(const ExpressionsDomain &cpu, const ExpressionsDomainGpu &gpu) {
    assert(gpu.nZerofiers() == cpu.nZerofiers() && gpu.size() == cpu.size() && gpu.extendBits() == cpu.extendBits());
    for (uint64_t b = 0; b < cpu.nZerofiers(); ++b) {
        const uint64_t mask = gpu.zerofierMasks()[b];
        const Column values = download(gpu.zerofiers()[b], mask + 1);
        for (uint64_t i = 0; i < cpu.size(); ++i) {
            assert(same(&values[i & mask], &cpu.zerofier(b)[i], 1));
        }
    }
}

// Every code block of `air` on H and on its parts, on the host through the tables and, with `gpu`,
// on the device: each the CPU's, values and errors.
void testAnAir(const Air &air, Random &random, bool gpu) {
    const Expressions cpu(air.bin, air.info);
    const OperandTypes types = air.bin.types();
    const OperandLayout layout =
        PilFflonk::operandLayout(air.bin, air.info.openingPoints.size(), air.info.boundaries.size());
    const uint64_t nBits = testBits(air);
    const uint64_t nBitsExt = nBits + (air.nBitsExt - air.info.nBits);
    const uint64_t NExt = uint64_t(1) << nBitsExt;
    const Values values(layout, types, NExt, random, gpu);

    // H, then the parts of each size, part by part.
    std::vector<uint64_t> sizes = {nBits};
    for (uint64_t partBits : {nBits, nBits + 1, nBitsExt}) {
        if (partBits <= nBitsExt && std::find(sizes.begin() + 1, sizes.end(), partBits) == sizes.end()) {
            sizes.push_back(partBits);
        }
    }
    std::vector<DeviceBuffer> workspaces;
    std::vector<std::vector<Domain>> domains(sizes.size());
    domains[0].push_back(traceDomain(nBits, gpu));
    for (uint64_t s = 1; s < sizes.size(); ++s) {
        for (uint64_t part = 0; part < (NExt >> sizes[s]); ++part) {
            domains[s].push_back(partDomain(nBits, nBitsExt, sizes[s], part, air.info.boundaries, workspaces, gpu));
            if (gpu) {
                expectTheCpusZerofiers(domains[s].back().cpu, domains[s].back().gpu[0]);
            }
        }
    }

    std::vector<std::unique_ptr<ExpressionsGpu>> devices;
    DeviceBuffer expressionsArgs, expressionsNumbers, constraintsArgs, constraintsNumbers, dest;
    if (gpu) {
        auto copy = [](const ParserArgs &code, DeviceBuffer &args, DeviceBuffer &numbers) {
            args = upload(code.args.data(), code.args.size() * sizeof(uint32_t));
            numbers = upload(code.numbers.data(), code.numbers.size() * sizeof(FrElement));
            return DeviceCode{reinterpret_cast<const uint32_t *>(args.data()),
                              reinterpret_cast<const FrElement *>(numbers.data())};
        };
        const DeviceCode e = copy(air.bin.expressionsBinArgsExpressions, expressionsArgs, expressionsNumbers);
        const DeviceCode c = copy(air.bin.expressionsBinArgsConstraints, constraintsArgs, constraintsNumbers);
        // The temporaries in shared memory, and in device memory.
        devices.emplace_back(new ExpressionsGpu(air.bin, air.info, e, c));
        devices.emplace_back(new ExpressionsGpu(air.bin, air.info, e, c, 0));
        dest = DeviceBuffer(NExt * sizeof(FrElement));
    }

    std::vector<uint64_t> table((layout.bytes + 7) / 8);
    uint8_t *tables = reinterpret_cast<uint8_t *>(table.data());
    uint64_t evaluated = 0, refused = 0;
    for (const Code &code : codesOf(air.bin)) {
        const ParserArgs &args =
            code.constraint ? air.bin.expressionsBinArgsConstraints : air.bin.expressionsBinArgsExpressions;
        for (uint64_t s = 0; s < sizes.size(); ++s) {
            const uint64_t size = domains[s][0].cpu.size(), nParts = s == 0 ? 1 : NExt / size;
            Column expected(size * nParts);
            std::string error;
            for (uint64_t part = 0; part < domains[s].size(); ++part) {
                const Domain &d = domains[s][part];
                Column out(size);
                error = outcome([&] {
                    if (code.constraint) {
                        cpu.calculateConstraint(code.id, d.cpu, values.host, out.data());
                    } else {
                        cpu.calculateExpression(code.id, d.cpu, values.host, out.data());
                    }
                });
                if (!error.empty()) {
                    break;
                }
                for (uint64_t i = 0; i < size; ++i) {
                    expected[part + nParts * i] = out[i];
                }
                const ParserParams &params =
                    code.constraint ? cpu.checkedConstraint(code.id, d.cpu.nZerofiers(), values.host, out.data())
                                    : cpu.checkedExpression(code.id, d.cpu.nZerofiers(), values.host, out.data());
                PilFflonk::encodeOperands(layout, types, values.host, args.numbers.data(),
                                          cpu.shifts(size, d.cpu.extendBits()), d.hostZerofiers, d.hostMasks, tables,
                                          tables);
                const OperandTables t = PilFflonk::operandTables(layout, types, tables, size);
                assert(same(throughTheTables(params, args, t, types.tmp(), size).data(), out.data(), size));
            }
            for (const auto &device : devices) {
                // No element: a value it does not write is not the CPU's.
                const std::vector<uint8_t> poison(size * nParts * sizeof(FrElement), 0xff);
                gpu_plonk_memcpy_h2d(dest.data(), poison.data(), poison.size());
                for (uint64_t part = 0; part < domains[s].size(); ++part) {
                    FrElement *at = reinterpret_cast<FrElement *>(dest.data()) + part;
                    const std::string got = outcome([&] {
                        if (code.constraint) {
                            device->calculateConstraint(code.id, domains[s][part].gpu[0], values.device, at, nParts);
                        } else {
                            device->calculateExpression(code.id, domains[s][part].gpu[0], values.device, at, nParts);
                        }
                    });
                    if (got != error) {
                        std::fprintf(stderr, "%s, %s %lu: the CPU says \"%s\" and the GPU \"%s\"\n", air.name.c_str(),
                                     code.constraint ? "constraint" : "expression", code.id, error.c_str(),
                                     got.c_str());
                        assert(false);
                    }
                    if (!error.empty()) {
                        break;
                    }
                }
                if (error.empty() &&
                    !same(download(dest.data(), size * nParts).data(), expected.data(), size * nParts)) {
                    std::fprintf(stderr, "%s, %s %lu on 2^%lu points: the GPU's values are not the CPU's\n",
                                 air.name.c_str(), code.constraint ? "constraint" : "expression", code.id,
                                 sizes[s]);
                    assert(false);
                }
            }
            (error.empty() ? evaluated : refused) += 1;
        }
    }
    std::printf("pilfflonk_test: %s: %lu code blocks on 2^%lu rows, extended to 2^%lu, %s: %lu of them on a "
                "domain size computed, %lu refused alike\n",
                air.name.c_str(), codesOf(air.bin).size(), nBits, nBitsExt,
                gpu ? "the device's and the tables' values the CPU's" : "the tables' values the CPU's", evaluated,
                refused);
}

// operandLayout of Sample.bin (sample() of setup/pilfflonk/tests/bytecode.rs, two stages): room
// for what its code reads in both sections, fixed column 0, columns 0 … 2 of stage 1 and 0 of
// stage 2, publics 0 and 1, air value 0, proof values 0 and 1, airgroup value 0 and challenge 0;
// and the tables encodeOperands writes, read back through operandAddress.
void testTheTablesOfTheSample() {
    const Air air = sampleAir();
    const OperandTypes types = air.bin.types();
    const OperandLayout layout = PilFflonk::operandLayout(air.bin, 5, 4);
    assert((layout.columnStart == std::vector<uint32_t>{0, 1, 4, 5, 5}));
    assert((layout.nScalars == std::vector<uint64_t>{2, 0, 1, 2, 1, 1}));
    assert(layout.nShifts == 5 && layout.nZerofiers == 4);
    assert(layout.values == 0 && layout.scalars == 7 * 32 && layout.columns == layout.scalars + 6 * 8);
    assert(layout.shifts == layout.columns + 5 * 8 && layout.zerofiers == layout.shifts + 5 * 8);
    assert(layout.masks == layout.zerofiers + 4 * 8 && layout.columnStarts == layout.masks + 4 * 8);
    assert(layout.bytes == layout.columnStarts + 5 * 4 + 4);

    // Distinct addresses for every column, Zi and number, to see which one each operand reads.
    Random random(60);
    ProverValues values;
    values.columns = {{reinterpret_cast<const FrElement *>(0x1000)},
                      {reinterpret_cast<const FrElement *>(0x2000), reinterpret_cast<const FrElement *>(0x3000)},
                      {reinterpret_cast<const FrElement *>(0x4000)}};
    values.publics = random.column(3); // more than the code reads
    values.airValues = random.column(1);
    values.proofValues = random.column(1); // fewer: the second stays 0
    values.airgroupValues = random.column(1);
    values.challenges = random.column(1);
    const void *numbers = reinterpret_cast<const void *>(0x5000);
    const std::vector<const void *> zerofiers = {reinterpret_cast<const void *>(0x6000),
                                                 reinterpret_cast<const void *>(0x7000)};
    std::vector<uint64_t> table((layout.bytes + 7) / 8);
    uint8_t *bytes = reinterpret_cast<uint8_t *>(table.data());
    PilFflonk::encodeOperands(layout, types, values, numbers, {3, 5, 0, 11, 13}, zerofiers, {1, 15}, bytes, bytes);
    const OperandTables t = PilFflonk::operandTables(layout, types, bytes, 16);
    assert(t.mask == 15 && t.ziType == types.zi() && t.scalarsType == types.publics());
    auto address = [&](uint32_t type, uint32_t arg1, uint32_t arg2, uint64_t i) {
        return reinterpret_cast<uint64_t>(PilFflonk::operandAddress(t, type, arg1, arg2, i));
    };
    // Columns: at opening point arg2, cyclically; the one past what the values have is null.
    assert(address(0, 0, 0, 0) == 0x1000 + 3 * 32 && address(0, 0, 1, 12) == 0x1000 + 1 * 32);
    assert(address(1, 1, 4, 7) == 0x3000 + 4 * 32 && address(2, 0, 3, 9) == 0x4000 + 4 * 32);
    assert(address(1, 2, 2, 0) == 0);
    // Zi: the everyRow's e = 2 values, the other's 16; boundaries the domain has not, null.
    assert(address(types.zi(), 1, 0, 13) == 0x6000 + 1 * 32 && address(types.zi(), 2, 0, 13) == 0x7000 + 13 * 32);
    assert(t.zerofiers[2] == nullptr && t.zerofiers[3] == nullptr);
    // Scalars: their values in the tables, the numbers the code's.
    auto value = [&](uint32_t type, uint32_t arg1) -> const FrElement & {
        return *static_cast<const FrElement *>(PilFflonk::operandAddress(t, type, arg1, 0, 5));
    };
    assert(address(types.numbers(), 2, 0, 5) == 0x5000 + 2 * 32);
    for (uint32_t i = 0; i < 2; ++i) {
        assert(same(&value(types.publics(), i), &values.publics[i], 1));
    }
    assert(same(&value(types.airValues(), 0), &values.airValues[0], 1));
    assert(same(&value(types.proofValues(), 0), &values.proofValues[0], 1));
    assert(E.fr.isZero(value(types.proofValues(), 1)));
    assert(same(&value(types.airgroupValues(), 0), &values.airgroupValues[0], 1));
    assert(same(&value(types.challenges(), 0), &values.challenges[0], 1));
}

// The device's Zi of a part, of every kind of boundary, is cosetPart's: on parts of fewer points
// than a warp, of one block and of many, everyFrames that exclude nothing, a few rows or all of
// them; and both refuse alike.
void testTheZerofiersOnTheDevice() {
    for (uint64_t nBits : {uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(5), uint64_t(8), uint64_t(11)}) {
        const uint64_t N = uint64_t(1) << nBits;
        std::vector<Boundary> boundaries = {{BoundaryType::EveryRow, 0, 0},
                                            {BoundaryType::FirstRow, 0, 0},
                                            {BoundaryType::LastRow, 0, 0},
                                            {BoundaryType::EveryFrame, 0, 0}};
        if (N >= 4) {
            boundaries.push_back({BoundaryType::EveryFrame, 1, 2});
            boundaries.push_back({BoundaryType::EveryFrame, 3, 0});
        }
        // Every row, on domains where the CPU's product of N factors per point is quick.
        if (nBits <= 5) {
            boundaries.push_back({BoundaryType::EveryFrame, N / 2, N / 2});
            boundaries.push_back({BoundaryType::EveryFrame, 1, N - 1});
        }
        for (uint64_t ext : {uint64_t(0), uint64_t(1), uint64_t(3)}) {
            const uint64_t nBitsExt = nBits + ext;
            for (uint64_t partBits = nBits; partBits <= nBitsExt; ++partBits) {
                for (uint64_t part = 0; part < (uint64_t(1) << (nBitsExt - partBits)); ++part) {
                    const DeviceBuffer workspace(ExpressionsDomainGpu::cosetPartBytes(nBits, partBits, boundaries));
                    const ExpressionsDomain cpu =
                        ExpressionsDomain::cosetPart(nBits, nBitsExt, partBits, part, boundaries);
                    const ExpressionsDomainGpu gpu =
                        ExpressionsDomainGpu::cosetPart(nBits, nBitsExt, partBits, part, boundaries, workspace.data());
                    assert(gpu.nBits() == nBits);
                    expectTheCpusZerofiers(cpu, gpu);
                }
            }
        }
    }
    const DeviceBuffer workspace(1024);
    const std::vector<Boundary> tooMany = {{BoundaryType::EveryFrame, 5, 4}};
    // nBits, nBitsExt, partBits and part.
    for (const std::array<uint64_t, 4> &a :
         std::vector<std::array<uint64_t, 4>>{{3, 5, 2, 0}, {3, 5, 6, 0}, {3, 5, 4, 2}, {3, 4, 3, 0}}) {
        const std::vector<Boundary> &boundaries = a[1] == 4 ? tooMany : std::vector<Boundary>{};
        const std::string expected = outcome([&] { ExpressionsDomain::cosetPart(a[0], a[1], a[2], a[3], boundaries); });
        assert(!expected.empty());
        assert(outcome([&] {
                   ExpressionsDomainGpu::cosetPart(a[0], a[1], a[2], a[3], boundaries, workspace.data());
               }) == expected);
    }
    assert(outcome([&] { ExpressionsDomainGpu::cosetPart(3, 4, 3, 0, {}, nullptr); }) ==
           "invalid_argument: ExpressionsDomainGpu::cosetPart: workspace is null");
    assert(outcome([] { ExpressionsDomainGpu::trace(29); }) == outcome([] { ExpressionsDomain::trace(29); }));
}

// What the device refuses besides the CPU's: a stride of 0, constraints without their code, and
// more shared memory than a block may take.
void testWhatTheDeviceRefuses() {
    const Air air = sampleAir();
    const ParserArgs &code = air.bin.expressionsBinArgsExpressions;
    const DeviceBuffer args = upload(code.args.data(), code.args.size() * sizeof(uint32_t));
    const DeviceBuffer numbers = upload(code.numbers.data(), code.numbers.size() * sizeof(FrElement));
    const DeviceCode e{reinterpret_cast<const uint32_t *>(args.data()),
                       reinterpret_cast<const FrElement *>(numbers.data())};
    const ExpressionsGpu gpu(air.bin, air.info, e);
    const ExpressionsDomainGpu trace = ExpressionsDomainGpu::trace(air.info.nBits);
    Random random(61);
    const Values values(PilFflonk::operandLayout(air.bin, 5, 4), air.bin.types(), 8, random, true);
    const DeviceBuffer dest(8 * sizeof(FrElement));
    FrElement *d = reinterpret_cast<FrElement *>(dest.data());
    assert(outcome([&] { gpu.calculateExpression(3, trace, values.device, d, 0); }) ==
           "invalid_argument: ExpressionsGpu::calculateExpression: a stride of 0");
    assert(outcome([&] { gpu.calculateConstraint(0, trace, values.device, d); }) ==
           "logic_error: ExpressionsGpu::calculateConstraint: the constraints' code is not on the device");
    assert(outcome([&] { gpu.calculateExpression(3, trace, values.device, nullptr); }) ==
           "invalid_argument: Expressions::calculateExpression: dest is null");
    assert(outcome([&] { gpu.calculateExpression(4, trace, values.device, d); }) ==
           outcome([&] { air.bin.expression(4); }));
    assert(outcome([&] {
               (void)ExpressionsGpu(air.bin, air.info, e, DeviceCode(), ExpressionsGpu::MAX_SHARED_BYTES + 1);
           }) == "invalid_argument: ExpressionsGpu: 49153 bytes of shared memory per block, and a block may take "
                 "49152");
}

// Q of the key at `dir` at its size, on every part of N and of 2N points, on the device and on the
// CPU, timed: the interpreter's kernel alone (its parts' Zi apart), and the CPU's calculateExpression.
void timeQ(const std::string &dir) {
    const std::vector<BytecodeFiles> found = bytecodesIn(dir);
    assert(found.size() == 1);
    const Air air = loadAir(found[0].first, found[0].second);
    const ParserParams &q = air.bin.expression(air.info.cExpId);
    const OperandTypes types = air.bin.types();
    const Expressions cpu(air.bin, air.info);
    const DeviceCode none;
    const DeviceBuffer args = upload(air.bin.expressionsBinArgsExpressions.args.data(),
                                     air.bin.expressionsBinArgsExpressions.args.size() * sizeof(uint32_t));
    const DeviceBuffer numbers = upload(air.bin.expressionsBinArgsExpressions.numbers.data(),
                                        air.bin.expressionsBinArgsExpressions.numbers.size() * sizeof(FrElement));
    const ExpressionsGpu gpu(air.bin, air.info,
                             DeviceCode{reinterpret_cast<const uint32_t *>(args.data()),
                                        reinterpret_cast<const FrElement *>(numbers.data())},
                             none);
    const std::vector<PilFflonk::ColumnRead> reads = PilFflonk::columnsRead(air.bin, air.info.cExpId);
    const uint64_t nBits = air.info.nBits, NExt = uint64_t(1) << air.nBitsExt;
    const DeviceBuffer dest(NExt * sizeof(FrElement));
    Random random(62);
    using Clock = std::chrono::steady_clock;
    auto seconds = [](Clock::duration d) { return std::chrono::duration<double>(d).count(); };
    for (uint64_t partBits : {nBits, nBits + 1}) {
        const uint64_t S = uint64_t(1) << partBits, nParts = NExt / S;
        // Q's columns on a part, and its scalars, random; the same on every part.
        ProverValues host, device;
        host.columns.resize(types.zi());
        device.columns.resize(types.zi());
        std::vector<Column> columns;
        std::vector<DeviceBuffer> deviceColumns;
        for (const PilFflonk::ColumnRead &c : reads) {
            columns.push_back(random.column(S));
            deviceColumns.push_back(upload(columns.back().data(), S * sizeof(FrElement)));
            host.columns[c.type].resize(std::max<size_t>(host.columns[c.type].size(), c.index + 1), nullptr);
            device.columns[c.type].resize(host.columns[c.type].size(), nullptr);
            host.columns[c.type][c.index] = columns.back().data();
            device.columns[c.type][c.index] = reinterpret_cast<const FrElement *>(deviceColumns.back().data());
        }
        randomScalars(PilFflonk::operandLayout(air.bin, air.info.openingPoints.size(), air.info.boundaries.size()),
                      types, random, host, device);

        Clock::duration onGpu{}, onCpu{};
        Column expected(NExt), part(S);
        const DeviceBuffer workspace(ExpressionsDomainGpu::cosetPartBytes(nBits, partBits, air.info.boundaries));
        for (uint64_t p = 0; p < nParts; ++p) {
            const ExpressionsDomainGpu domain = ExpressionsDomainGpu::cosetPart(nBits, air.nBitsExt, partBits, p,
                                                                                air.info.boundaries, workspace.data());
            gpu_plonk_cuda_device_sync();
            const Clock::time_point start = Clock::now();
            gpu.calculateExpression(air.info.cExpId, domain, device, reinterpret_cast<FrElement *>(dest.data()) + p,
                                    nParts);
            gpu_plonk_cuda_device_sync();
            onGpu += Clock::now() - start;

            const Clock::time_point cpuStart = Clock::now();
            const ExpressionsDomain cpuDomain =
                ExpressionsDomain::cosetPart(nBits, air.nBitsExt, partBits, p, air.info.boundaries);
            cpu.calculateExpression(air.info.cExpId, cpuDomain, host, part.data());
            onCpu += Clock::now() - cpuStart;
            for (uint64_t i = 0; i < S; ++i) {
                expected[p + nParts * i] = part[i];
            }
        }
        const bool equal = same(download(dest.data(), NExt).data(), expected.data(), NExt);
        std::printf("pilfflonk_test: timing: Q of %s (%u ops, %u temporaries, %zu columns) on %lu parts of 2^%lu "
                    "points: %.4f s on the GPU (the kernel), %.4f s on the CPU (with its Zi), %s\n",
                    air.name.c_str(), q.nOps, q.nTemp, reads.size(), nParts, partBits, seconds(onGpu), seconds(onCpu),
                    equal ? "byte for byte" : "DIFFERENT");
        assert(equal);
    }
}

} // namespace

#endif

void runExpressionsGpuTests() {
#ifdef __USE_CUDA__
    const bool gpu = gpuUnderTest("the interpreter on the device");
    testTheTablesOfTheSample();
    if (gpu) {
        testTheZerofiersOnTheDevice();
        testWhatTheDeviceRefuses();
    }
    Random random(60);
    testAnAir(sampleAir(), random, gpu);
    testAnAir(loadAir(fixture("fibonacci/Fibonacci.bin"), fixture("fibonacci/Fibonacci.pilfflonkinfo.json")), random,
              gpu);
    testAnAir(loadAir(fixture("sum_bus/SumBus.bin"), fixture("sum_bus/SumBus.pilfflonkinfo.json")), random, gpu);
    for (const auto &[bin, info] : keysUnderTest()) {
        testAnAir(loadAir(bin, info), random, gpu);
    }
    if (const char *key = std::getenv("PILFFLONK_GPU_TIMING_KEY")) {
        if (gpu) {
            timeQ(key);
        }
    }
#endif
}

} // namespace PilFflonkTest
