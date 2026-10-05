// Tests of the columns of the stages after the first on the device (pilfflonk_hints_gpu.hpp), in a
// library built with the GPU, byte for byte against the CPU's code:
// - the kernels of pilfflonk_hints_kernels.hpp: the quotient of a hint from every kind of operand
//   (columns at every shift, numbers, the numerator in place), the first row of a zero denominator,
//   and the running sum, with PLONK's running product as the hints call it, on sizes around a warp,
//   a block and the scans' levels, the scans' work exactly what they say;
// - the sum bus of setup/pilfflonk/tests/fixtures/bytecode/sum_bus/ on a key on the GPU: its
//   columns of stage 2 the oracle's, the CPU key's and check's; a denominator 0 refused with the
//   CPU's message on the first row, of an expression and of a column; what stage 2 copies (the
//   blinding factors up, the committed polynomials and a row per hint down, and no column); and
//   Instance::column before and after Q;
// - every provingKey/ under the directories of PILFFLONK_GPU_HINT_KEYS (colon-separated: the bus
//   fixtures, the wrap's layouts, the bench keys), with a random witness from a fixed seed and random
//   challenges: every column of its later stages on the device is the one check computes on the host,
//   or both refuse it with the same message; on a key of 2^16 rows or fewer, the CPU key's
//   commitments and Q's outcome are the GPU key's, and a denominator made a witness column that is
//   0 on two rows is refused on both with the same message.
// The scratch of a stage in the arena (StageScratch) is checked without a GPU too, and so are the
// keys under test, on the CPU alone: their columns from commitStage, check's.
#include "pilfflonk_test.hpp"
#include "pilfflonk_test_ptau.hpp"

#ifdef __USE_CUDA__

#include <limits.h>
#include <unistd.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pilfflonk_error.hpp"
#include "pilfflonk_hints_gpu.hpp"
#include "pilfflonk_hints_kernels.hpp"
#include "pilfflonk_key_gpu.hpp"
#include "pilfflonk_prover.hpp"
#include "pilfflonk_proving_key.hpp"
#include "pilfflonk_rng.hpp"
#include "pilfflonk_srs.hpp"
#include "pilfflonk_transcript.hpp"

// The PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu).
extern "C" void gpu_plonk_memcpy_h2d(void *dst, const void *src, size_t bytes);
extern "C" void gpu_plonk_memcpy_d2h(void *dst, const void *src, size_t bytes);
extern "C" void gpu_plonk_prefix_scan_multiply(void *dData, uint64_t N, void *dWork);

#endif

namespace PilFflonkTest {

#ifdef __USE_CUDA__

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using PilFflonk::AirKey;
using PilFflonk::BlindingRng;
using PilFflonk::ColumnRead;
using PilFflonk::DeviceBuffer;
using PilFflonk::Device;
using PilFflonk::ExpressionsBin;
using PilFflonk::FrElement;
using PilFflonk::G1Point;
using PilFflonk::GlobalInfo;
using PilFflonk::HintFieldValue;
using PilFflonk::HintInput;
using PilFflonk::HintOp;
using PilFflonk::Instance;
using PilFflonk::PilfflonkInfo;
using PilFflonk::ProvingKey;
using PilFflonk::Srs;
using PilFflonk::StdHint;
using PilFflonk::UnsatisfiedError;
using Engine = AltBn128::Engine;
using Column = std::vector<FrElement>;

Engine &E = Engine::engine;

constexpr uint64_t NO_ROW = UINT64_MAX;
// Around a warp, a block of 256 threads and powers of two, and the scans' levels: one block of 1024,
// two levels from 1025, three from 2^20 + 1.
const std::vector<uint64_t> SIZES = {1,    2,    31,    32,    33,    255,   256,   257,  1023,
                                     1024, 1025, 4095,  4097,  65535, 65537, (uint64_t(1) << 20) + 1,
                                     uint64_t(1) << 21};
// The keys whose CPU key is compared too, and whose denominators are changed: of 2^16 rows at most.
constexpr uint64_t SMALL_KEY_BITS = 16;
// The keys under test are of 2^22 rows at most: the host's check of a larger one takes too long.
constexpr uint64_t LARGEST_KEY_BITS = 22;

// Elements from a fixed seed: 0, 1, r − 1 and three equal ones first, then below 2^253 < r as
// Montgomery limbs.
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

    FrElement nonZero() {
        FrElement e = element();
        return E.fr.isZero(e) ? E.fr.one() : e;
    }

    Column column(uint64_t n) {
        const FrElement same = element();
        const FrElement edges[] = {E.fr.zero(), E.fr.one(), E.fr.negOne(), same, same, same};
        Column c(n);
        for (uint64_t i = 0; i < n; ++i) {
            c[i] = i < 6 ? edges[i] : element();
        }
        return c;
    }

    // A column with no zero, its edges 1 and r − 1.
    Column nonZeroColumn(uint64_t n) {
        Column c = column(n);
        for (FrElement &e : c) {
            if (E.fr.isZero(e)) {
                e = nonZero();
            }
        }
        return c;
    }

    uint64_t below(uint64_t n) { return generator() % n; }

private:
    std::mt19937_64 generator;
};

DeviceBuffer upload(const void *data, uint64_t bytes) {
    DeviceBuffer device(std::max<uint64_t>(bytes, 1));
    gpu_plonk_memcpy_h2d(device.data(), data, bytes);
    return device;
}

DeviceBuffer upload(const Column &c) { return upload(c.data(), c.size() * sizeof(FrElement)); }

Column download(const void *device, uint64_t n) {
    Column host(n);
    gpu_plonk_memcpy_d2h(host.data(), device, n * sizeof(FrElement));
    return host;
}

uint64_t downloadRow(const DeviceBuffer &row) {
    uint64_t value = 0;
    gpu_plonk_memcpy_d2h(&value, row.data(), sizeof(value));
    return value;
}

bool same(const FrElement *a, const FrElement *b, uint64_t n) { return std::memcmp(a, b, n * sizeof(FrElement)) == 0; }

bool same(const Column &a, const Column &b) { return a.size() == b.size() && same(a.data(), b.data(), a.size()); }

bool samePoint(G1Point a, G1Point b) {
    Engine::G1PointAffine x, y;
    E.g1.copy(x, a);
    E.g1.copy(y, b);
    return std::memcmp(&x, &y, sizeof(x)) == 0;
}

bool contains(const std::string &s, const std::string &part) { return s.find(part) != std::string::npos; }

FrElement inverse(const FrElement &a) {
    FrElement r;
    E.fr.inv(r, a);
    return r;
}

std::vector<uint8_t> readBytes(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    assert(file);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::string readText(const std::string &path) {
    const std::vector<uint8_t> bytes = readBytes(path);
    return std::string(bytes.begin(), bytes.end());
}

// What `call` throws as an UnsatisfiedError, or "" if it throws nothing.
template <typename Call>
std::string unsatisfied(Call call) {
    try {
        call();
    } catch (const UnsatisfiedError &e) {
        return e.what();
    }
    return "";
}

template <typename Call>
std::string invalidArgument(Call call) {
    try {
        call();
    } catch (const std::invalid_argument &e) {
        return e.what();
    }
    assert(!"expected std::invalid_argument");
    return "";
}

// ---------------------------------------------------------------------------------------------
// The kernels
// ---------------------------------------------------------------------------------------------

HintOperand columnOperand(const DeviceBuffer &values, uint64_t shift) {
    HintOperand op{};
    op.values = values.data();
    op.shift = shift;
    return op;
}

HintOperand numberOperand(const FrElement &number) {
    HintOperand op{};
    std::memcpy(op.number, &number, sizeof(op.number));
    return op;
}

// An operand on the host, as the kernel reads it: values[(i + shift) mod n], or the number.
struct HostOperand {
    const Column *values; // null: the number
    uint64_t shift;
    FrElement number;

    FrElement at(uint64_t i, uint64_t n) const { return values == nullptr ? number : (*values)[(i + shift) % n]; }
};

// The quotient Instance::computeHintColumns computes, and the first row of a zero denominator.
Column expectedQuotient(const HostOperand &num, const HostOperand &den, uint64_t n, uint64_t &firstZero) {
    Column out(n);
    firstZero = NO_ROW;
    for (uint64_t i = 0; i < n; ++i) {
        const FrElement d = den.at(i, n);
        if (E.fr.isZero(d)) {
            firstZero = std::min(firstZero, i);
            out[i] = E.fr.zero();
            continue;
        }
        E.fr.mul(out[i], num.at(i, n), inverse(d));
    }
    return out;
}

// pilfflonk_gpu_hint_quotient on n = 2^bits rows, n from 1 to 2^20: of a column and a column at the
// shifts 0, 1, n − 1 and a random one; of the numerator in place over dest and a number; of a number
// and a column that is 0 on two rows, the first one found whatever the shift; and of a denominator
// that is the number 0, whose first row is 0.
void testHintQuotient(Random &random) {
    for (uint64_t bits : {0, 1, 3, 5, 8, 10, 12, 16, 20}) {
        const uint64_t n = uint64_t(1) << bits;
        const Column num = random.column(n);
        const Column den = random.nonZeroColumn(n);
        const DeviceBuffer dNum = upload(num), dDen = upload(den), dest(n * sizeof(FrElement)),
                           row(sizeof(uint64_t));
        auto run = [&](const HintOperand &a, const HintOperand &b, const HostOperand &ha,
                       const HostOperand &hb) {
            pilfflonk_gpu_hint_quotient(dest.data(), n, &a, &b, reinterpret_cast<uint64_t *>(row.data()));
            uint64_t firstZero = 0;
            const Column expected = expectedQuotient(ha, hb, n, firstZero);
            assert(same(download(dest.data(), n), expected));
            assert(downloadRow(row) == firstZero);
            return firstZero;
        };
        for (uint64_t s1 : {uint64_t(0), uint64_t(1), n - 1, random.below(n)}) {
            for (uint64_t s2 : {uint64_t(0), uint64_t(1), n - 1, random.below(n)}) {
                assert(run(columnOperand(dNum, s1), columnOperand(dDen, s2), {&num, s1, {}}, {&den, s2, {}}) == NO_ROW);
            }
        }
        // In place: dest holds the numerator.
        gpu_plonk_memcpy_h2d(dest.data(), num.data(), n * sizeof(FrElement));
        const FrElement number = random.nonZero();
        HintOperand inPlace{};
        inPlace.values = dest.data();
        assert(run(inPlace, numberOperand(number), {&num, 0, {}}, {nullptr, 0, number}) == NO_ROW);
        // Zeros on two rows, the later one first in memory for half of the shifts.
        Column zeros = den;
        const uint64_t a = random.below(n), b = random.below(n);
        zeros[a] = E.fr.zero();
        zeros[b] = E.fr.zero();
        const DeviceBuffer dZeros = upload(zeros);
        for (uint64_t s : {uint64_t(0), n - 1, random.below(n)}) {
            const uint64_t first = run(numberOperand(number), columnOperand(dZeros, s), {nullptr, 0, number},
                                       {&zeros, s, {}});
            assert(first == std::min((a + n - s) % n, (b + n - s) % n));
        }
        assert(run(columnOperand(dNum, 0), numberOperand(E.fr.zero()), {&num, 0, {}}, {nullptr, 0, E.fr.zero()}) == 0);
    }
}

// PLONK's running product (gpu_plonk_prefix_scan_multiply) and the running sum
// (pilfflonk_gpu_prefix_scan_add) of every size of SIZES are the CPU's, row after row, and write no
// more work than pilfflonk_gpu_prefix_scan_work_elements says.
void testPrefixScans(Random &random) {
    assert(pilfflonk_gpu_prefix_scan_work_elements(1) == 0 && pilfflonk_gpu_prefix_scan_work_elements(1024) == 0);
    assert(pilfflonk_gpu_prefix_scan_work_elements(1025) == 2);
    assert(pilfflonk_gpu_prefix_scan_work_elements((uint64_t(1) << 20) + 1) == 1025 + 2);
    constexpr uint64_t GUARD = 64;
    for (uint64_t n : SIZES) {
        const Column data = random.nonZeroColumn(n);
        Column products(n), sums(n);
        for (uint64_t i = 0; i < n; ++i) {
            products[i] = data[i];
            sums[i] = data[i];
            if (i > 0) {
                E.fr.mul(products[i], products[i], products[i - 1]);
                E.fr.add(sums[i], sums[i], sums[i - 1]);
            }
        }
        const uint64_t workElements = pilfflonk_gpu_prefix_scan_work_elements(n);
        const Column guard = random.column(workElements + GUARD);
        for (bool product : {true, false}) {
            const DeviceBuffer d = upload(data), work = upload(guard);
            if (product) {
                gpu_plonk_prefix_scan_multiply(d.data(), n, work.data());
            } else {
                pilfflonk_gpu_prefix_scan_add(d.data(), n, work.data());
            }
            assert(same(download(d.data(), n), product ? products : sums));
            const Column after = download(work.data(), workElements + GUARD);
            assert(same(after.data() + workElements, guard.data() + workElements, GUARD));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------------------------

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

std::string busFixture(const std::string &name) {
    return repoPath("setup/pilfflonk/tests/fixtures/bytecode/sum_bus/" + name);
}

// The files of a key of one AIR, its .bin parsed so that a test may change it.
struct AirFiles {
    std::string name;   // the AIR's
    std::string global; // the globalInfo's text
    std::string srs;    // the path of the SRS
    std::string info;   // the pilfflonkinfo's text
    std::vector<uint8_t> bin;
    std::vector<uint8_t> constants;
};

// A key of those files on `device`, its .bin changed by `change`.
std::unique_ptr<ProvingKey> keyOf(const AirFiles &files, Device device,
                                  const std::function<void(ExpressionsBin &)> &change = nullptr) {
    ExpressionsBin bin = ExpressionsBin::parse(files.bin.data(), files.bin.size(), files.name + ".bin");
    if (change) {
        change(bin);
    }
    std::shared_ptr<PilFflonk::GpuKey> gpu;
    if (device == Device::Gpu) {
        gpu = std::make_shared<PilFflonk::GpuKey>(Srs::load(files.srs));
    }
    std::vector<std::vector<std::unique_ptr<AirKey>>> airs(1);
    airs[0].push_back(std::make_unique<AirKey>(PilfflonkInfo::parse(files.info), std::move(bin), files.constants.data(),
                                               files.constants.size(), files.name, gpu.get()));
    return std::make_unique<ProvingKey>(GlobalInfo::parse(files.global), Srs::load(files.srs), std::move(airs),
                                        std::move(gpu));
}

// AIR airId of airgroup airgroupId of the provingKey/ at dir (pilfflonk/docs/formats.md#provingkey).
AirFiles airFilesOf(const std::string &dir, uint64_t airgroupId, uint64_t airId) {
    const GlobalInfo global = GlobalInfo::load(dir + "/pilout.globalInfo.json");
    const std::string air = global.airs[airgroupId][airId].name;
    const std::string base =
        dir + "/" + global.name + "/" + global.airGroups[airgroupId] + "/airs/" + air + "/air/" + air;
    return AirFiles{air,
                    readText(dir + "/pilout.globalInfo.json"),
                    dir + "/" + global.name + "/pilfflonk/pilfflonk.srs.bin",
                    readText(base + ".pilfflonkinfo.json"),
                    readBytes(base + ".bin"),
                    readBytes(base + ".const")};
}

// The sum bus's fixture as a key of its own, with an SRS of the test ptau.
struct SumBus {
    TestDir dir;
    AirFiles files;
    json oracle = json::parse(readText(busFixture("SumBus.oracle.json")));
    std::vector<uint8_t> witness = readBytes(busFixture("SumBus.witness.bin"));

    SumBus() {
        const std::string ptau = dir.file("sum_bus.ptau");
        writeTestPtau(ptau, 512);
        files.srs = dir.file("sum_bus.srs.bin");
        Srs::fromPtau(ptau, 512).save(files.srs);
        files.name = "SumBus";
        files.global = "{\"name\": \"sum\", \"airs\": [[{\"name\": \"SumBus\", \"num_rows\": 32}]], \"air_groups\": "
                       "[\"SumBus\"], \"aggTypes\": [[]], \"backend\": \"pilfflonk\", \"formatVersion\": 1, \"field\": "
                       "\"bn128\", \"nPublics\": 1, \"numChallenges\": [0], \"numProofValues\": [], "
                       "\"proofValuesMap\": [], \"publicsMap\": []}";
        files.info = readText(busFixture("SumBus.pilfflonkinfo.json"));
        files.bin = readBytes(busFixture("SumBus.bin"));
        files.constants = readBytes(busFixture("SumBus.const"));
    }

    std::vector<FrElement> values(const json &decimals) const {
        std::vector<FrElement> out;
        for (const json &v : decimals) {
            FrElement e;
            E.fr.fromString(e, v.get<std::string>(), 10);
            out.push_back(e);
        }
        return out;
    }
};

constexpr uint64_t SUM_BUS_N = 32;
constexpr uint64_t SUM_BUS_COLUMNS = 3; // a, b, mul
constexpr uint64_t SUM_BUS_A = 0, SUM_BUS_B = 1;

FrElement traceValue(const std::vector<uint8_t> &trace, uint64_t row, uint64_t column, uint64_t nColumns) {
    FrElement v;
    E.fr.fromRprLE(v, trace.data() + (row * nColumns + column) * 32, 32);
    return v;
}

void setTraceValue(std::vector<uint8_t> &trace, uint64_t row, uint64_t column, uint64_t nColumns,
                   const FrElement &v) {
    PilFflonk::encodeFr(v, trace.data() + (row * nColumns + column) * 32);
}

std::unique_ptr<Instance> instanceOf(const ProvingKey &pk, const std::vector<uint8_t> &trace,
                                     const std::vector<FrElement> &publics, uint8_t seed) {
    uint8_t bytes[32] = {seed};
    return std::make_unique<Instance>(pk, 0, 0, trace.data(), trace.size(), std::vector<FrElement>{}, publics,
                                      std::vector<FrElement>{}, std::make_unique<BlindingRng>(bytes));
}

// The index in openingPoints of the row offset 0.
uint64_t ownRow(const PilfflonkInfo &info) {
    const auto at = std::find(info.openingPoints.begin(), info.openingPoints.end(), int64_t(0));
    assert(at != info.openingPoints.end());
    return static_cast<uint64_t>(at - info.openingPoints.begin());
}

// The denominator of hint h of `bin` made the committed column cmId at its own row.
void denominatorIsColumn(ExpressionsBin &bin, const PilfflonkInfo &info, const StdHint &hint, uint64_t cmId) {
    const std::string field = hint.kind == StdHint::Kind::ImCol ? "denominator" : "denominator_air";
    for (PilFflonk::HintField &f : bin.hints[hint.hint].fields) {
        if (f.name == field) {
            HintFieldValue &v = f.values[0];
            v.op = HintOp::Cm;
            v.id = cmId;
            v.rowOffsetIndex = ownRow(info);
            return;
        }
    }
    assert(!"the hint has no denominator");
}

// ---------------------------------------------------------------------------------------------
// The sum bus
// ---------------------------------------------------------------------------------------------

// The scratch of the sum bus's stage 2 in the arena: from the work buffer on, with the fixed columns
// its im_col's denominator reads, and its stages' phase holding it.
void testTheScratch() {
    const SumBus bus;
    const std::unique_ptr<ProvingKey> pk = keyOf(bus.files, Device::Cpu);
    const AirKey &air = pk->air(0, 0);
    const PilFflonk::StageScratch scratch = PilFflonk::stageScratch(air, 2);
    assert(!scratch.fixed.empty() && std::is_sorted(scratch.fixed.begin(), scratch.fixed.end()));
    for (const StdHint &hint : air.stdHints()) {
        for (const HintInput *in : {&hint.numerator, &hint.denominator}) {
            if (in->kind == HintInput::Kind::Expression) {
                for (const ColumnRead &c : PilFflonk::columnsRead(air.bin(), in->expId)) {
                    assert(c.type != 0 ||
                           std::find(scratch.fixed.begin(), scratch.fixed.end(), c.index) != scratch.fixed.end());
                }
            }
        }
    }
    const uint64_t N = air.n();
    assert(scratch.fixedValues == 0 && scratch.denominator >= scratch.fixed.size() * N * sizeof(FrElement));
    assert(scratch.work >= scratch.denominator + N * sizeof(FrElement) && scratch.zeroRow >= scratch.work);
    assert(scratch.bytes >= scratch.zeroRow + sizeof(uint64_t));
    // Stage 1 has no hints: its scratch is the fixed columns its im pols read.
    const PilFflonk::StageScratch first = PilFflonk::stageScratch(air, 1);
    assert(first.bytes == ((first.fixed.size() * N * sizeof(FrElement) + 255) & ~uint64_t(255)));
    const uint64_t most = std::max(scratch.bytes, first.bytes);
    assert(PilFflonk::stageScratchBytes(air) == most);
    const PilFflonk::ArenaLayout layout = PilFflonk::arenaLayout(air);
    assert(layout.hints == layout.work && layout.hintBytes == most);
    assert(layout.stageBytes >= layout.hints + layout.hintBytes && layout.bytes >= layout.stageBytes);
}

// On a key on the GPU, the sum bus's columns of stage 2 (gsum, im_single and the im pol) are the
// oracle's, check's and a CPU key's; stage 2 copies to the device its blinding factors only, and to
// the host the counts of its committed polynomials' coefficients and a row per hint; Instance::column
// copies a column the first time it is asked for, and refuses one it has not once commitQ has begun;
// and the proof's commitments are the CPU key's.
void testTheSumBusOnTheGpu() {
    const SumBus bus;
    const std::unique_ptr<ProvingKey> cpu = keyOf(bus.files, Device::Cpu), gpu = keyOf(bus.files, Device::Gpu);
    const std::vector<FrElement> publics = bus.values(bus.oracle["publics"]);
    const std::vector<FrElement> challenges = bus.values(bus.oracle["challenges"]);
    const AirKey &air = gpu->air(0, 0);
    const uint64_t N = air.n();
    assert(N == SUM_BUS_N);
    const json &stage2 = bus.oracle["stage2"];
    std::vector<G1Point> onCpu;
    {
        const std::unique_ptr<Instance> inst = instanceOf(*cpu, bus.witness, publics, 3);
        onCpu = inst->commitStage(1, {});
        const std::vector<G1Point> two = inst->commitStage(2, challenges);
        onCpu.insert(onCpu.end(), two.begin(), two.end());
    }
    const std::unique_ptr<Instance> inst = instanceOf(*gpu, bus.witness, publics, 3);
    const std::vector<std::vector<FrElement>> checked = inst->checkColumns(challenges);
    std::vector<G1Point> onGpu = inst->commitStage(1, {});
    const PilFflonk::CopyVolume &volume = gpu->gpuKey()->copies();
    const PilFflonk::CopyVolume::Totals before = volume.totals();
    const std::vector<G1Point> two = inst->commitStage(2, challenges);
    const PilFflonk::CopyVolume::Totals after = volume.totals();
    onGpu.insert(onGpu.end(), two.begin(), two.end());
    assert(onGpu.size() == onCpu.size());
    for (uint64_t i = 0; i < onGpu.size(); ++i) {
        assert(samePoint(onGpu[i], onCpu[i]));
    }
    uint64_t up = 0, down = 0;
    for (uint64_t f = 0; f < air.info().layout.size(); ++f) {
        if (air.info().layout[f].stage == 2) {
            const uint64_t k = air.info().layout[f].k, b = air.blindLength(f);
            up += k * b * sizeof(FrElement);
            down += k * sizeof(uint64_t);
        }
    }
    down += std::count_if(air.stdHints().begin(), air.stdHints().end(), [](const StdHint &h) { return h.stage == 2; }) *
            sizeof(uint64_t);
    assert(after.toDevice - before.toDevice == up && after.toHost - before.toHost == down);

    assert(stage2.size() == air.cmIds()[2].size());
    for (uint64_t p = 0; p < stage2.size(); ++p) {
        const std::vector<FrElement> oracle = bus.values(stage2[p]);
        assert(same(inst->column(2, p), oracle.data(), N));
        assert(same(inst->column(2, p), checked[2].data() + p * N, N));
    }
    // Any std_vc: the witness satisfies every constraint.
    assert(unsatisfied([&] { inst->commitQ({E.fr.one()}); }).empty());
    for (uint64_t p = 0; p < stage2.size(); ++p) {
        assert(same(inst->column(2, p), bus.values(stage2[p]).data(), N));
    }
}

// Once commitQ has begun, a column not copied before is refused, of stage 2 or of stage 1 (whose
// columns are on the device too), and so are check's columns if check had not run before; those
// copied before are still there.
void testAColumnAfterQ() {
    const SumBus bus;
    const std::unique_ptr<ProvingKey> gpu = keyOf(bus.files, Device::Gpu);
    const std::vector<FrElement> publics = bus.values(bus.oracle["publics"]);
    const std::vector<FrElement> challenges = bus.values(bus.oracle["challenges"]);
    const uint64_t N = SUM_BUS_N;
    {
        const std::unique_ptr<Instance> inst = instanceOf(*gpu, bus.witness, publics, 4);
        inst->commitStage(1, {});
        inst->commitStage(2, challenges);
        assert(same(inst->column(2, 1), bus.values(bus.oracle["stage2"][1]).data(), N));
        const std::vector<std::vector<FrElement>> checked = inst->checkColumns(challenges);
        assert(same(inst->column(1, SUM_BUS_B), checked[1].data() + SUM_BUS_B * N, N));
        assert(unsatisfied([&] { inst->commitQ({E.fr.one()}); }).empty());
        assert(contains(invalidArgument([&] { inst->column(2, 0); }),
                        "Instance::column: on a key on the GPU, the columns of stage 2 are on the device until Q is "
                        "committed"));
        assert(contains(invalidArgument([&] { inst->column(1, SUM_BUS_A); }),
                        "Instance::column: on a key on the GPU, the columns of stage 1 are on the device until Q is "
                        "committed"));
        assert(same(inst->column(2, 1), bus.values(bus.oracle["stage2"][1]).data(), N));
        assert(same(inst->column(1, SUM_BUS_B), checked[1].data() + SUM_BUS_B * N, N));
        assert(same(inst->checkColumns(challenges)[1].data(), checked[1].data(), checked[1].size()));
    }
    const std::unique_ptr<Instance> late = instanceOf(*gpu, bus.witness, publics, 4);
    late->commitStage(1, {});
    late->commitStage(2, challenges);
    assert(unsatisfied([&] { late->commitQ({E.fr.one()}); }).empty());
    assert(contains(invalidArgument([&] { late->checkColumns(challenges); }),
                    "Instance::check: on a key on the GPU, the witness columns are on the device until Q is "
                    "committed"));
}

// A denominator 0 on rows of the sum bus, refused on the GPU with the CPU's message, of the first
// row, by check and by commitStage(2), and the stage left uncommitted: gsum_col's, an expression of
// a and b, made 0 at rows 20 and 5; and im_col's made the column a, 0 at row 2 (the generator's
// witness), and with a 0 at rows 17 and 9 only.
void testTheSumBusZeroDenominators() {
    const SumBus bus;
    const std::vector<FrElement> publics = bus.values(bus.oracle["publics"]);
    const std::vector<FrElement> challenges = bus.values(bus.oracle["challenges"]);
    const FrElement &alpha = challenges[0], &gamma = challenges[1];
    std::vector<uint8_t> trace = bus.witness;
    for (uint64_t row : {uint64_t(20), uint64_t(5)}) {
        // 1 + a·α + b·α² + γ = 0 (the lookup's term of gsum_col).
        FrElement e, alpha2;
        E.fr.mul(alpha2, alpha, alpha);
        E.fr.mul(e, traceValue(trace, row, SUM_BUS_B, SUM_BUS_COLUMNS), alpha2);
        E.fr.add(e, e, E.fr.one());
        E.fr.add(e, e, gamma);
        E.fr.neg(e, e);
        FrElement a;
        E.fr.mul(a, e, inverse(alpha));
        setTraceValue(trace, row, SUM_BUS_A, SUM_BUS_COLUMNS, a);
    }
    std::vector<uint8_t> moved = bus.witness;
    for (uint64_t row = 0; row < SUM_BUS_N; ++row) {
        if (E.fr.isZero(traceValue(moved, row, SUM_BUS_A, SUM_BUS_COLUMNS))) {
            setTraceValue(moved, row, SUM_BUS_A, SUM_BUS_COLUMNS, E.fr.one());
        }
    }
    setTraceValue(moved, 17, SUM_BUS_A, SUM_BUS_COLUMNS, E.fr.zero());
    setTraceValue(moved, 9, SUM_BUS_A, SUM_BUS_COLUMNS, E.fr.zero());
    struct Case {
        bool imColOfA;
        const std::vector<uint8_t> *trace;
        std::string message;
    };
    const std::vector<Case> cases = {
        {false, &trace, "SumBus: the denominator of hint 1 (gsum_col, column gsum) is 0 at row 5"},
        {true, &bus.witness, "SumBus: the denominator of hint 0 (im_col, column im_single) is 0 at row 2"},
        {true, &moved, "SumBus: the denominator of hint 0 (im_col, column im_single) is 0 at row 9"},
    };
    for (const Case &c : cases) {
        std::function<void(ExpressionsBin &)> change;
        if (c.imColOfA) {
            const std::unique_ptr<ProvingKey> plain = keyOf(bus.files, Device::Cpu);
            const AirKey &air = plain->air(0, 0);
            const StdHint imCol = air.stdHints()[0];
            const PilfflonkInfo info = air.info();
            const uint64_t a = air.cmIds()[1][air.witnessColumns()[SUM_BUS_A]];
            assert(imCol.kind == StdHint::Kind::ImCol);
            change = [=](ExpressionsBin &bin) { denominatorIsColumn(bin, info, imCol, a); };
        }
        std::string messages[2];
        for (int d = 0; d < 2; ++d) {
            const std::unique_ptr<ProvingKey> pk = keyOf(bus.files, d == 0 ? Device::Cpu : Device::Gpu, change);
            const std::unique_ptr<Instance> inst = instanceOf(*pk, *c.trace, publics, 5);
            const std::string checked = unsatisfied([&] { inst->checkColumns(challenges); });
            inst->commitStage(1, {});
            messages[d] = unsatisfied([&] { inst->commitStage(2, challenges); });
            assert(checked == messages[d] && inst->nextStage() == 2);
        }
        assert(contains(messages[0], c.message) && messages[1] == messages[0]);
    }
}

// ---------------------------------------------------------------------------------------------
// The keys under test
// ---------------------------------------------------------------------------------------------

// Every pilfflonk provingKey/ (a directory with a pilout.globalInfo.json of pilfflonk's,
// GlobalInfo::load) under the directories of PILFFLONK_GPU_HINT_KEYS, in order; another key is
// skipped, and said so.
std::vector<std::string> keysUnderTest() {
    std::vector<std::string> keys;
    const char *list = std::getenv("PILFFLONK_GPU_HINT_KEYS");
    std::string dirs = list != nullptr ? list : "";
    while (!dirs.empty()) {
        const size_t colon = dirs.find(':');
        const std::string dir = dirs.substr(0, colon);
        std::vector<std::string> found;
        for (const auto &entry : fs::recursive_directory_iterator(dir)) {
            if (entry.path().filename() != "pilout.globalInfo.json") {
                continue;
            }
            try {
                GlobalInfo::load(entry.path().string());
                found.push_back(entry.path().parent_path().string());
            } catch (const PilFflonk::FormatError &e) {
                std::printf("pilfflonk_test: %s is not a pilfflonk provingKey/, skipped: %s\n",
                            entry.path().parent_path().c_str(), e.what());
            }
        }
        assert(!found.empty() && "a directory under test has no provingKey/");
        std::sort(found.begin(), found.end());
        keys.insert(keys.end(), found.begin(), found.end());
        dirs = colon == std::string::npos ? "" : dirs.substr(colon + 1);
    }
    return keys;
}

// A random witness of `air`: N rows of its C stage-1 columns, as canonical scalars, with `zeroColumn`
// (if below C) 0 on `zeroRows`.
std::vector<uint8_t> randomWitness(const AirKey &air, Random &random, uint64_t zeroColumn = UINT64_MAX,
                                   const std::vector<uint64_t> &zeroRows = {}) {
    const uint64_t N = air.n(), C = air.witnessColumns().size();
    std::vector<uint8_t> trace(N * C * PilFflonk::FR_BYTES);
    for (uint64_t v = 0; v < N * C; ++v) {
        FrElement canonical;
        E.fr.fromMontgomery(canonical, random.element());
        std::memcpy(trace.data() + v * PilFflonk::FR_BYTES, canonical.v, PilFflonk::FR_BYTES);
    }
    for (uint64_t row : zeroRows) {
        if (zeroColumn < C) {
            std::memset(trace.data() + (row * C + zeroColumn) * PilFflonk::FR_BYTES, 0, PilFflonk::FR_BYTES);
        }
    }
    return trace;
}

// Random challenges of stages 2 … nStages of `inst`, by stage (index s − 2) and then stageId.
std::vector<std::vector<FrElement>> randomChallenges(const Instance &inst, Random &random) {
    std::vector<std::vector<FrElement>> out;
    for (uint64_t s = 2; s <= inst.air().info().nStages; ++s) {
        std::vector<FrElement> c;
        for (uint64_t k = 0; k < inst.nChallenges(s); ++k) {
            c.push_back(random.nonZero());
        }
        out.push_back(c);
    }
    return out;
}

std::vector<FrElement> flat(const std::vector<std::vector<FrElement>> &byStage) {
    std::vector<FrElement> out;
    for (const std::vector<FrElement> &c : byStage) {
        out.insert(out.end(), c.begin(), c.end());
    }
    return out;
}

// What the stages of `inst` give: their commitments, or the message of the refusal of the first one
// refused; and, with `columns`, the device's columns of every stage s >= 2, or of those before the
// refused one.
struct Stages {
    std::vector<G1Point> commitments;
    std::string refused;
    std::vector<std::vector<FrElement>> columns; // by stage
};

Stages commitStages(Instance &inst, const std::vector<std::vector<FrElement>> &challenges, bool columns) {
    Stages out;
    const AirKey &air = inst.air();
    out.columns.resize(air.info().nStages + 1);
    for (uint64_t s = 1; s <= air.info().nStages; ++s) {
        const std::vector<FrElement> given = s == 1 ? std::vector<FrElement>{} : challenges[s - 2];
        std::vector<G1Point> points;
        out.refused = unsatisfied([&] { points = inst.commitStage(s, given); });
        if (!out.refused.empty()) {
            assert(inst.nextStage() == s);
            return out;
        }
        out.commitments.insert(out.commitments.end(), points.begin(), points.end());
        for (uint64_t p = 0; columns && s > 1 && p < air.cmIds()[s].size(); ++p) {
            const FrElement *c = inst.column(s, p);
            out.columns[s].insert(out.columns[s].end(), c, c + air.n());
        }
    }
    return out;
}

// One key under test (see the file's comment), on the GPU with `gpu`; without, on the CPU alone, as
// a check of the keys and of the host's side of the test. Counts in `compared` the AIRs whose
// columns were compared, and in `refusedAlike` those both refused.
void testAKey(const std::string &dir, Random &random, bool gpu, uint64_t &compared, uint64_t &refusedAlike) {
    const std::unique_ptr<ProvingKey> key = ProvingKey::load(dir, gpu ? Device::Gpu : Device::Cpu);
    const GlobalInfo &global = key->globalInfo();
    const std::vector<Device> devices =
        gpu ? std::vector<Device>{Device::Cpu, Device::Gpu} : std::vector<Device>{Device::Cpu};
    for (uint64_t g = 0; g < global.airs.size(); ++g) {
        for (uint64_t a = 0; a < global.airs[g].size(); ++a) {
            const AirKey &air = key->air(g, a);
            const PilfflonkInfo &info = air.info();
            if (info.nStages < 2 || info.nBits > LARGEST_KEY_BITS) {
                std::printf("pilfflonk_test: %s: %s has %" PRIu64 " stage(s) of 2^%" PRIu64 " rows, skipped\n",
                            dir.c_str(), air.name().c_str(), info.nStages, info.nBits);
                continue;
            }
            const uint64_t N = air.n();
            const std::vector<FrElement> publics(global.nPublics, E.fr.one());
            const std::vector<uint8_t> trace = randomWitness(air, random);
            std::vector<std::vector<FrElement>> challenges;
            std::string checkRefused;
            std::vector<std::vector<FrElement>> checked;
            Stages committed;
            {
                const std::unique_ptr<Instance> inst = instanceOf(*key, trace, publics, 7);
                challenges = randomChallenges(*inst, random);
                checkRefused = unsatisfied([&] { checked = inst->checkColumns(flat(challenges)); });
                committed = commitStages(*inst, challenges, true);
            }
            assert(committed.refused == checkRefused);
            if (committed.refused.empty()) {
                for (uint64_t s = 2; s <= info.nStages; ++s) {
                    assert(committed.columns[s].size() == checked[s].size() && same(committed.columns[s], checked[s]));
                }
                ++compared;
            } else {
                ++refusedAlike;
            }
            std::printf("pilfflonk_test: %s: %s, 2^%" PRIu64 " rows, %zu hints: the %s's columns of its later stages "
                        "%s\n",
                        dir.c_str(), air.name().c_str(), info.nBits, air.stdHints().size(), gpu ? "device" : "host",
                        committed.refused.empty() ? "are check's" : ("refused as check: " + committed.refused).c_str());
            if (info.nBits > SMALL_KEY_BITS || global.airs.size() != 1 || global.airs[0].size() != 1) {
                continue;
            }

            // The CPU key's commitments and Q, and the GPU key's.
            const AirFiles files = airFilesOf(dir, g, a);
            std::vector<std::string> q(devices.size());
            std::vector<Stages> stages(devices.size());
            const FrElement vc = random.nonZero();
            for (uint64_t d = 0; d < devices.size(); ++d) {
                const std::unique_ptr<ProvingKey> pk = keyOf(files, devices[d]);
                const std::unique_ptr<Instance> inst = instanceOf(*pk, trace, publics, 9);
                stages[d] = commitStages(*inst, challenges, false);
                if (stages[d].refused.empty()) {
                    q[d] = unsatisfied([&] { inst->commitQ({vc}); });
                }
                assert(stages[d].refused == stages[0].refused && q[d] == q[0]);
                assert(stages[d].commitments.size() == stages[0].commitments.size());
                for (uint64_t i = 0; i < stages[0].commitments.size(); ++i) {
                    assert(samePoint(stages[d].commitments[i], stages[0].commitments[i]));
                }
            }

            // A denominator a witness column, 0 on two rows, of its first and last hint of stage 2.
            std::vector<StdHint> hints;
            for (const StdHint &h : air.stdHints()) {
                if (h.stage == 2) {
                    hints.push_back(h);
                }
            }
            for (const StdHint *hint : {&hints.front(), &hints.back()}) {
                const uint64_t cmId = air.cmIds()[1][air.witnessColumns()[0]];
                const std::vector<uint64_t> rows = {N - 1 - random.below(N / 2), random.below(N / 2)};
                const std::vector<uint8_t> zeroed = randomWitness(air, random, 0, rows);
                std::vector<std::string> messages(devices.size());
                for (uint64_t d = 0; d < devices.size(); ++d) {
                    const std::unique_ptr<ProvingKey> pk = keyOf(
                        files, devices[d], [&](ExpressionsBin &bin) { denominatorIsColumn(bin, info, *hint, cmId); });
                    const std::unique_ptr<Instance> inst = instanceOf(*pk, zeroed, publics, 11);
                    inst->commitStage(1, {});
                    messages[d] = unsatisfied([&] { inst->commitStage(2, challenges[0]); });
                    assert(inst->nextStage() == 2 && !messages[d].empty() && messages[d] == messages[0]);
                }
                std::printf("pilfflonk_test: %s: hint %" PRIu64 " of a zero denominator: %s\n", dir.c_str(),
                            hint->hint, messages.back().c_str());
            }
        }
    }
}

void testTheKeysUnderTest(Random &random, bool gpu) {
    uint64_t compared = 0, refusedAlike = 0;
    for (const std::string &dir : keysUnderTest()) {
        testAKey(dir, random, gpu, compared, refusedAlike);
    }
    std::printf("pilfflonk_test: the stages' columns on the %s: %" PRIu64 " AIRs the CPU's, %" PRIu64
                " refused alike\n",
                gpu ? "device" : "host", compared, refusedAlike);
}

} // namespace

#endif

void runHintsGpuTests() {
#ifdef __USE_CUDA__
    testTheScratch();
    Random random(61);
    if (!gpuUnderTest("the stages' columns on the device")) {
        testTheKeysUnderTest(random, false);
        return;
    }
    testHintQuotient(random);
    testPrefixScans(random);
    testTheSumBusOnTheGpu();
    testAColumnAfterQ();
    testTheSumBusZeroDenominators();
    testTheKeysUnderTest(random, true);
#endif
}

} // namespace PilFflonkTest
