// The device side of a ProvingKey on the GPU (pilfflonk_key_gpu.hpp): compiled into the GPU library
// only (the Makefile's %_gpu.cpp rule), with g++. It calls the kernels of pilfflonk_kernels.cu and
// the PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu) through their C linkage.
#include "pilfflonk_key_gpu.hpp"

#include <omp.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "pilfflonk_expressions_gpu.hpp"
#include "pilfflonk_hints_gpu.hpp"
#include "pilfflonk_kernels.hpp"
#include "pilfflonk_lde_gpu.hpp"
#include "pilfflonk_opening_gpu.hpp"
#include "pilfflonk_proving_key.hpp"
#include "thread_utils.hpp"
#include "timer.hpp"

// The PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu), declared as plonk_prover_gpu.c.cuh
// declares them.
extern "C" void gpu_plonk_cuda_malloc(void **dBuffer, uint64_t buffeSize);
extern "C" void gpu_plonk_cuda_free(void *dBuffer);
extern "C" void gpu_plonk_cuda_malloc_pinned_buffer(void **pinnedBuffer, size_t pinnedSize);
extern "C" void gpu_plonk_free_pinned_buffer(void *pinnedBuffer);
extern "C" void gpu_plonk_set_device(int gpuId);
extern "C" void *gpu_plonk_create_cuda_stream_nonblocking();
extern "C" void gpu_plonk_destroy_cuda_stream(void *stream);
extern "C" void gpu_plonk_sync_cuda_stream(void *stream);
extern "C" void gpu_plonk_memcpy_h2d_async(void *dst, const void *src, size_t bytes, void *stream);
extern "C" void gpu_plonk_pin_host_memory(void *ptr, size_t bytes);
extern "C" void gpu_plonk_unpin_host_memory(void *ptr);
extern "C" void gpu_plonk_precompute_omega_tables_async(void *dBases, void *dTid, const void *omega4xPtr,
                                                        uint32_t blockSize, uint32_t numBlocks, void *stream);

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

// Gpu's device.
constexpr int DEVICE = 0;
// Each half of the Staging's pinned buffer.
constexpr uint64_t STAGING_HALF_BYTES = uint64_t(32) << 20;
// What a proof allocates besides the arena that gpuBudget does not count element by element: the
// allocator's rounding of each allocation to its pages, and sppark's NTT tables, of a few MB.
constexpr uint64_t MARGIN_BYTES = uint64_t(256) << 20;
// sppark's MSM: the bits of a BN254 scalar, and a bucket of its xyzz_t<fp_t>::mem_t.
constexpr uint64_t SCALAR_BITS = 254;
constexpr uint64_t BUCKET_BYTES = 4 * sizeof(FrElement);

uint64_t aligned(uint64_t bytes) { return (bytes + 255) & ~uint64_t(255); }

uint64_t log2Floor(uint64_t x) {
    uint64_t bits = 0;
    while (x >>= 1) {
        ++bits;
    }
    return bits;
}

// Regions carved one after another from offset 0, each 256-byte aligned.
class Carver {
public:
    uint64_t take(uint64_t bytes) {
        const uint64_t start = end;
        end = aligned(end + bytes);
        return start;
    }
    uint64_t size() const { return end; }

private:
    uint64_t end = 0;
};

// The largest degree bound of the layout's f: the longest MSM of a proof.
uint64_t largestDegree(const AirKey &air) {
    uint64_t largest = 0;
    for (const LayoutEntry &f : air.info().layout) {
        largest = std::max<uint64_t>(largest, f.degree);
    }
    return largest;
}

// Where GpuAirKey keeps what is on the device while the key lives, as offsets in its buffer.
struct ResidentLayout {
    uint64_t fixed, ops, args, numbers, positions, offsets;
    uint64_t tableLength; // entries of the offsets' table: the polynomials of every f but Q's, then
                          // one per fixed column
    uint64_t bytes;
};

ResidentLayout residentLayout(const AirKey &air) {
    const PilfflonkInfo &info = air.info();
    const ParserArgs &code = air.bin().expressionsBinArgsExpressions;
    ResidentLayout r{};
    r.tableLength = info.nConstants;
    for (const LayoutEntry &f : info.layout) {
        if (f.stage != info.qStage()) {
            r.tableLength += f.k;
        }
    }
    Carver carver;
    r.fixed = carver.take(info.nConstants * air.n() * sizeof(FrElement));
    r.ops = carver.take(code.ops.size() * sizeof(uint8_t));
    r.args = carver.take(code.args.size() * sizeof(uint32_t));
    r.numbers = carver.take(code.numbers.size() * sizeof(FrElement));
    r.positions = carver.take(air.witnessColumns().size() * sizeof(uint64_t));
    r.offsets = carver.take(r.tableLength * sizeof(uint64_t));
    r.bytes = carver.size();
    return r;
}

// `stagePos` (each once) as runs of consecutive ones, (first, count), in increasing order.
GpuAirKey::ColumnRuns runsOf(std::vector<uint64_t> stagePos) {
    std::sort(stagePos.begin(), stagePos.end());
    GpuAirKey::ColumnRuns runs;
    for (uint64_t p : stagePos) {
        if (!runs.empty() && runs.back().first + runs.back().second == p) {
            ++runs.back().second;
        } else {
            runs.emplace_back(p, 1);
        }
    }
    return runs;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// DeviceBuffer, Staging
// ---------------------------------------------------------------------------------------------

DeviceBuffer::DeviceBuffer(uint64_t _bytes) : bytes(_bytes) {
    if (bytes > 0) {
        void *allocated = nullptr;
        gpu_plonk_cuda_malloc(&allocated, bytes);
        memory = static_cast<uint8_t *>(allocated);
    }
}

DeviceBuffer::~DeviceBuffer() { gpu_plonk_cuda_free(memory); }

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : memory(std::exchange(other.memory, nullptr)), bytes(std::exchange(other.bytes, 0)) {}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
        gpu_plonk_cuda_free(memory);
        memory = std::exchange(other.memory, nullptr);
        bytes = std::exchange(other.bytes, 0);
    }
    return *this;
}

Staging::Staging(uint64_t halfBytes, CopyVolume &volume) : half(halfBytes), copies(volume) {
    gpu_plonk_cuda_malloc_pinned_buffer(&pinned, 2 * half);
    stream = gpu_plonk_create_cuda_stream_nonblocking();
    ready = pilfflonk_gpu_event_create();
    done[0] = pilfflonk_gpu_event_create();
    done[1] = pilfflonk_gpu_event_create();
}

Staging::~Staging() {
    gpu_plonk_sync_cuda_stream(stream);
    pilfflonk_gpu_event_destroy(done[1]);
    pilfflonk_gpu_event_destroy(done[0]);
    pilfflonk_gpu_event_destroy(ready);
    gpu_plonk_destroy_cuda_stream(stream);
    gpu_plonk_free_pinned_buffer(pinned);
}

void Staging::afterDefaultStream() {
    pilfflonk_gpu_event_record(ready, nullptr);
    pilfflonk_gpu_stream_wait_event(stream, ready);
}

void Staging::toDevice(void *dst, const void *src, uint64_t bytes) {
    if (bytes == 0) {
        return;
    }
    afterDefaultStream();
    uint8_t *halves[2] = {static_cast<uint8_t *>(pinned), static_cast<uint8_t *>(pinned) + half};
    uint64_t chunk = 0;
    for (uint64_t at = 0; at < bytes; at += half, ++chunk) {
        const uint64_t n = std::min(half, bytes - at);
        const uint64_t h = chunk & 1;
        // The half's previous copy is done before it is written again.
        pilfflonk_gpu_event_sync(done[h]);
        ThreadUtils::parcpy(halves[h], static_cast<const uint8_t *>(src) + at, n, omp_get_max_threads());
        gpu_plonk_memcpy_h2d_async(static_cast<uint8_t *>(dst) + at, halves[h], n, stream);
        pilfflonk_gpu_event_record(done[h], stream);
    }
    // The stream is in order: the last copy's event is every copy's.
    pilfflonk_gpu_stream_wait_event(nullptr, done[(chunk - 1) & 1]);
    copies.addToDevice(bytes);
}

void Staging::toHost(void *dst, const void *src, uint64_t bytes) {
    if (bytes == 0) {
        return;
    }
    afterDefaultStream();
    uint8_t *halves[2] = {static_cast<uint8_t *>(pinned), static_cast<uint8_t *>(pinned) + half};
    const uint64_t chunks = (bytes + half - 1) / half;
    auto length = [&](uint64_t chunk) { return std::min(half, bytes - chunk * half); };
    auto issue = [&](uint64_t chunk) {
        pilfflonk_gpu_memcpy_d2h_async(halves[chunk & 1], static_cast<const uint8_t *>(src) + chunk * half,
                                       length(chunk), stream);
        pilfflonk_gpu_event_record(done[chunk & 1], stream);
    };
    issue(0);
    for (uint64_t chunk = 0; chunk < chunks; ++chunk) {
        // The next chunk goes to the other half, whose last chunk is copied out already.
        if (chunk + 1 < chunks) {
            issue(chunk + 1);
        }
        pilfflonk_gpu_event_sync(done[chunk & 1]);
        ThreadUtils::parcpy(static_cast<uint8_t *>(dst) + chunk * half, halves[chunk & 1], length(chunk),
                            omp_get_max_threads());
    }
    copies.addToHost(bytes);
}

void Staging::toRegisteredHost(void *dst, const void *src, uint64_t bytes) {
    if (bytes == 0) {
        return;
    }
    afterDefaultStream();
    pilfflonk_gpu_memcpy_d2h_async(dst, src, bytes, stream);
    copies.addToHost(bytes);
}

void Staging::wait() { gpu_plonk_sync_cuda_stream(stream); }

// ---------------------------------------------------------------------------------------------
// Budgets
// ---------------------------------------------------------------------------------------------

ArenaLayout arenaLayout(const AirKey &air) {
    const PilfflonkInfo &info = air.info();
    const uint64_t N = air.n();
    ArenaLayout a;
    a.slot.assign(info.layout.size(), ArenaLayout::NONE);
    a.evaluations.assign(info.nStages + 1, 0);
    std::vector<uint64_t> factors(info.nStages + 1, 0), polys(info.nStages + 1, 0);
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        const LayoutEntry &entry = info.layout[f];
        if (entry.stage < 1 || entry.stage > info.nStages) {
            continue;
        }
        const uint64_t b = air.blindLength(f);
        a.slot[f] = a.polyElements;
        a.polyElements += entry.k * (N + b);
        factors[entry.stage] += entry.k * b;
        polys[entry.stage] += entry.k;
    }
    const uint64_t largest = largestDegree(air);
    a.workElements = std::max(N * air.witnessColumns().size(), largest);
    a.factorElements = *std::max_element(factors.begin(), factors.end());
    a.nCounts = std::max(*std::max_element(polys.begin(), polys.end()), info.nConstants);

    Carver carver;
    a.polys = carver.take(a.polyElements * sizeof(FrElement));
    const uint64_t polysEnd = carver.size();
    for (uint64_t s = 1; s <= info.nStages; ++s) {
        a.evaluations[s] = carver.take(air.cmIds()[s].size() * N * sizeof(FrElement));
    }
    a.work = carver.take(a.workElements * sizeof(FrElement));
    a.factors = carver.take(a.factorElements * sizeof(FrElement));
    a.counts = carver.take(a.nCounts * sizeof(uint64_t));
    a.stageBytes = carver.size();
    a.hints = a.work;
    a.hintBytes = stageScratchBytes(air);
    a.stageBytes = std::max(a.stageBytes, aligned(a.hints + a.hintBytes));

    uint64_t pieces = 0;
    for (uint64_t c : air.degrees().qPieceCoefficients) {
        pieces += c;
    }
    a.q = polysEnd;
    a.qElements = air.lde().extendedSize() + N * air.qReads().size();
    a.qTables = aligned(a.q + a.qElements * sizeof(FrElement));
    const uint64_t q = a.qTables + ldeTableElements(air.lde()) * sizeof(FrElement);
    a.qPieces = polysEnd;
    a.shplonk = aligned(a.qPieces + pieces * sizeof(FrElement));
    const uint64_t opening = a.shplonk + shplonkWorkspaceBytes(shplonkBounds(air));
    a.bytes = std::max({a.stageBytes, aligned(q), aligned(opening)});
    return a;
}

uint64_t componentOffset(const AirKey &air, const ArenaLayout &layout, uint64_t f, uint64_t j) {
    const LayoutEntry &entry = air.info().layout[f];
    return entry.stage == 0 ? entry.pols[j].id * air.n() : layout.slot[f] + j * (air.n() + air.blindLength(f));
}

uint64_t spparkMsmBytes(uint64_t n, uint32_t multiprocessors) {
    if (n == 0) {
        return 0;
    }
    // msm_t(nullptr, n): the window, the buckets and their histogram.
    const uint64_t rounded = (n + 31) & ~uint64_t(31);
    uint64_t wbits = 10;
    if (rounded > 192) {
        wbits = std::max<uint64_t>(10, std::min<uint64_t>(log2Floor(rounded + rounded / 2) - 8, 18));
    }
    const uint64_t nwins = (SCALAR_BITS - 1) / wbits + 1;
    const uint64_t row = uint64_t(1) << (wbits - 1);
    const uint64_t buckets = nwins * row + uint64_t(multiprocessors) * 256 / 32;
    const uint64_t blob = buckets * BUCKET_BYTES + nwins * row * sizeof(uint32_t);
    // invoke with the points and the scalars on the device: the digits and temporaries of a batch.
    const uint64_t lgn = log2Floor(n + n / 2);
    const uint64_t batch = std::max<uint64_t>(1, (uint64_t(1) << (std::max(lgn, wbits) - wbits)) >> 6);
    const uint64_t stride = (((n + batch - 1) / batch) + 31) & ~uint64_t(31);
    return blob + stride * 2 * 2 * sizeof(uint32_t) + nwins * stride * sizeof(uint32_t);
}

GpuBudget gpuBudget(const AirKey &air, uint32_t multiprocessors) {
    const uint64_t largest = largestDegree(air);
    GpuBudget budget;
    budget.resident = residentLayout(air).bytes + ExpressionsGpu::deviceBytesOf(air.bin(), air.info(), multiprocessors);
    budget.arena = arenaLayout(air).bytes;
    budget.transient = spparkMsmBytes(largest, multiprocessors) +
                       std::max(air.lde().extendedSize(), largest) * sizeof(FrElement) + MARGIN_BYTES;
    return budget;
}

void requireDeviceMemory(const std::string &what, uint64_t needed, uint64_t available) {
    if (needed > available) {
        throw std::invalid_argument("not enough GPU memory for " + what + ": it needs " + std::to_string(needed) +
                                    " bytes of device memory, and the device has " + std::to_string(available) +
                                    " free (pilfflonk/docs/performance.md#selection-memory-and-errors)");
    }
}

void logCopies(const CopyVolume &volume, const std::string &phase, const CopyVolume::Totals &since) {
    const CopyVolume::Totals now = volume.totals();
    zklog.trace("PILFFLONK_COPIES_" + phase + ": " + std::to_string(now.toDevice - since.toDevice) +
                " bytes to the device, " + std::to_string(now.toHost - since.toHost) + " to the host");
}

CopyLog::CopyLog(const GpuKey *_key, std::string phase)
    : key(_key), name(std::move(phase)), start(key != nullptr ? key->copies().totals() : CopyVolume::Totals()) {}

CopyLog::~CopyLog() {
    if (key != nullptr) {
        logCopies(key->copies(), name, start);
    }
}

std::unique_ptr<Poly> mirrorPolynomial(FrElement *coefs, uint64_t length, uint64_t deviceCount,
                                       const std::string &what) {
    std::unique_ptr<Poly> poly(Poly::fromReservedBuffer(Engine::engine, coefs, length));
    const uint64_t degree = deviceCount == 0 ? 0 : deviceCount - 1;
    if (poly->getDegree() != degree) {
        throw std::logic_error(what + ": its copy on the host has degree " + std::to_string(poly->getDegree()) +
                               ", and the GPU found " + std::to_string(degree) +
                               ": the copy is not what the device holds");
    }
    return poly;
}

// ---------------------------------------------------------------------------------------------
// GpuKey
// ---------------------------------------------------------------------------------------------

GpuKey::RegisteredHost::RegisteredHost(uint64_t _n) : elements(new FrElement[_n]), n(_n) {
    // Its pages first touched by every thread, which is faster than by the registration alone.
    ThreadUtils::parset(elements.get(), 0, n * sizeof(FrElement), omp_get_max_threads());
    gpu_plonk_pin_host_memory(elements.get(), n * sizeof(FrElement));
}

GpuKey::RegisteredHost::~RegisteredHost() { gpu_plonk_unpin_host_memory(elements.get()); }

GpuKey::GpuKey(const Srs &srs, GpuKeyOptions _options) : options(_options) {
    if (!Gpu::available()) {
        throw std::invalid_argument("GpuKey: no GPU: CUDA sees no device of compute capability 7.0 or above (or no "
                                    "driver)");
    }
    gpu_plonk_set_device(DEVICE);
    smCount = pilfflonk_gpu_multiprocessors();
    const uint64_t n = srs.nG1();
    // Powers up to h^n: the MSMs are of n scalars at most.
    const uint64_t nBlocks = (n >> 8) + 1;
    const uint64_t powersBytes = n * sizeof(G1PointAffine);
    const uint64_t tablesBytes = (nBlocks + 256) * sizeof(FrElement) + 2 * sizeof(uint64_t);
    requireDeviceMemory("the SRS's " + std::to_string(n) + " powers [τ^i]₁ and the tables of the MSMs' shift",
                        powersBytes + tablesBytes, available());
    transforms = std::make_unique<Gpu>(&srs.g1(0), n, &volume);
    tables = DeviceBuffer(tablesBytes);
    hBlocks = tables.data();
    hPowers = tables.data() + nBlocks * sizeof(FrElement);
    zeroOffset = reinterpret_cast<uint64_t *>(tables.data() + (nBlocks + 256) * sizeof(FrElement));
    count = zeroOffset + 1;
    gpu_plonk_precompute_omega_tables_async(tables.data(), tables.data() + nBlocks * sizeof(FrElement),
                                            &msmShiftRatio(), 256, static_cast<uint32_t>(nBlocks), nullptr);
    pilfflonk_gpu_memset_zero(zeroOffset, sizeof(uint64_t));
    transfers = std::make_unique<Staging>(STAGING_HALF_BYTES, volume);
    held = powersBytes + tablesBytes;
}

GpuKey::~GpuKey() = default;

uint64_t GpuKey::available() const {
    uint64_t free = 0, total = 0;
    pilfflonk_gpu_memory(&free, &total);
    if (options.memoryLimit != 0) {
        free = std::min(free, options.memoryLimit > held ? options.memoryLimit - held : 0);
    }
    return free;
}

void GpuKey::reserve(const std::string &air, const GpuBudget &budget, uint64_t mirrorElements, uint64_t qElements) {
    if (options.arena != nullptr && budget.arena > options.arenaBytes) {
        throw std::invalid_argument(air + ": a proof on the GPU needs an arena of " + std::to_string(budget.arena) +
                                    " bytes, and the one given has " + std::to_string(options.arenaBytes));
    }
    const uint64_t growth = options.arena == nullptr && budget.arena > arenaBytes ? budget.arena - arenaBytes : 0;
    transient = std::max(transient, budget.transient);
    requireDeviceMemory(air + " (its fixed columns, bytecode and tables, " + std::to_string(budget.resident) +
                            " bytes; the arena's growth to its proofs', " + std::to_string(growth) +
                            "; and what a proof allocates besides, " + std::to_string(transient) + ")",
                        budget.resident + growth + transient, available());
    if (growth > 0) {
        owned = DeviceBuffer();
        owned = DeviceBuffer(budget.arena);
        arenaBytes = budget.arena;
        held += growth;
    }
    if (mirrorElements > (mirrorBuffer != nullptr ? mirrorBuffer->n : 0)) {
        mirrorBuffer.reset();
        mirrorBuffer = std::make_unique<RegisteredHost>(mirrorElements);
        hostMirror = mirrorBuffer->elements.get();
    }
    qHost(qElements);
    held += budget.resident;
}

FrElement *GpuKey::qHost(uint64_t n) const {
    if (qBuffer == nullptr || n > qBuffer->n) {
        qBuffer.reset();
        qBuffer = std::make_unique<RegisteredHost>(n);
    }
    return qBuffer->elements.get();
}

void GpuKey::addShiftSum(uint64_t n, void *work) {
    if (shiftSums.count(n) != 0) {
        return;
    }
    pilfflonk_gpu_pack_shift(work, n, nullptr, nullptr, 0, 0, hBlocks, hPowers);
    G1Point sum = msmOnDevice(transforms->devicePowers(), work, n);
    if (Engine::engine.g1.isZero(sum)) {
        throw std::runtime_error("GpuKey: the GPU's MSM of the shift of " + std::to_string(n) +
                                 " points gave the point at infinity: msm_bn128_gpu_dev_ptr failed (or τ is a root "
                                 "of the shift's polynomial)");
    }
    shiftSums.emplace(n, sum);
}

bool GpuKey::allZero(const void *scalars, uint64_t n) const {
    pilfflonk_gpu_count_coefficients(count, scalars, zeroOffset, 1, n);
    uint64_t found = 0;
    transfers->toHost(&found, count, sizeof(found));
    return found == 0;
}

G1Point GpuKey::commit(const void *base, const uint64_t *offsets, uint64_t k, uint64_t length, uint64_t n,
                       void *work) const {
    const auto shiftSum = shiftSums.find(n);
    if (shiftSum == shiftSums.end()) {
        throw std::logic_error("GpuKey::commit: no shift sum for MSMs of " + std::to_string(n) + " scalars");
    }
    pilfflonk_gpu_pack_shift(work, n, base, offsets, k, length, hBlocks, hPowers);
    G1Point shifted = msmOnDevice(transforms->devicePowers(), work, n);
    Engine &E = Engine::engine;
    if (E.g1.isZero(shifted) && !allZero(work, n)) {
        throw std::runtime_error("GpuKey::commit: the GPU's MSM of " + std::to_string(n) +
                                 " points gave the point at infinity for shifted scalars that are not all zero: "
                                 "msm_bn128_gpu_dev_ptr failed (or τ is a root of the shifted polynomial)");
    }
    G1Point sum = shiftSum->second, result;
    E.g1.sub(result, shifted, sum);
    return result;
}

GpuKey::Lease::Lease(const GpuKey &key, const char *function) : owner(key) {
    std::unique_lock<std::mutex> guard(owner.leaseLock);
    if (owner.leased && owner.holder == std::this_thread::get_id()) {
        throw std::invalid_argument(std::string(function) +
                                    ": this thread proves already with another instance of the key, which holds "
                                    "the GPU memory of one proof at a time");
    }
    owner.leaseFree.wait(guard, [this] { return !owner.leased; });
    owner.leased = true;
    owner.holder = std::this_thread::get_id();
}

GpuKey::Lease::~Lease() {
    {
        const std::lock_guard<std::mutex> guard(owner.leaseLock);
        owner.leased = false;
        owner.holder = std::thread::id();
    }
    owner.leaseFree.notify_one();
}

// ---------------------------------------------------------------------------------------------
// GpuAirKey
// ---------------------------------------------------------------------------------------------

GpuAirKey::GpuAirKey(GpuKey &_key, const AirKey &_air, FrElement *fixedCoefs,
                     std::vector<std::unique_ptr<Poly>> &fixedPolys)
    : key(_key), air(_air), layout(arenaLayout(_air)) {
    const PilfflonkInfo &info = air.info();
    const uint64_t N = air.n();
    checkSrsFits(air, key.nPowers());
    key.reserve(air.name(), gpuBudget(air, key.multiprocessors()), layout.polyElements, layout.qElements);

    const ResidentLayout r = residentLayout(air);
    resident = DeviceBuffer(r.bytes);
    fixedOffset = r.fixed;
    opsOffset = r.ops;
    argsOffset = r.args;
    numbersOffset = r.numbers;
    positionsOffset = r.positions;
    offsetsOffset = r.offsets;

    // The offsets of the polynomials each f packs: of a fixed f, its columns' in the fixed
    // coefficients; of a stage's, its slots in the arena.
    std::vector<uint64_t> table;
    tableStart.assign(info.layout.size(), ArenaLayout::NONE);
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        const LayoutEntry &entry = info.layout[f];
        if (entry.stage == info.qStage()) {
            continue;
        }
        tableStart[f] = table.size();
        for (uint64_t j = 0; j < entry.k; ++j) {
            table.push_back(componentOffset(air, layout, f, j));
        }
    }
    fixedColumnsStart = table.size();
    for (uint64_t c = 0; c < info.nConstants; ++c) {
        table.push_back(c * N);
    }
    if (table.size() != r.tableLength) {
        throw std::logic_error("GpuAirKey: " + air.name() + "'s table of offsets has " + std::to_string(table.size()) +
                               " entries, and its layout " + std::to_string(r.tableLength));
    }
    const ParserArgs &code = air.bin().expressionsBinArgsExpressions;
    Staging &staging = key.staging();
    staging.toDevice(resident.data() + opsOffset, code.ops.data(), code.ops.size() * sizeof(uint8_t));
    staging.toDevice(resident.data() + argsOffset, code.args.data(), code.args.size() * sizeof(uint32_t));
    staging.toDevice(resident.data() + numbersOffset, code.numbers.data(), code.numbers.size() * sizeof(FrElement));
    staging.toDevice(resident.data() + positionsOffset, air.witnessColumns().data(),
                     air.witnessColumns().size() * sizeof(uint64_t));
    staging.toDevice(resident.data() + offsetsOffset, table.data(), table.size() * sizeof(uint64_t));

    // What the host computes of each stage: stage 1's im pols. The device computes every column of
    // the later ones (computeStageColumns).
    computedOnHost.assign(info.nStages + 1, {});
    for (uint64_t s = 1; s <= info.nStages; ++s) {
        std::vector<uint64_t> stagePos;
        for (uint64_t p = 0; p < air.cmIds()[s].size(); ++p) {
            if (s == 1 && info.cmPolsMap[air.cmIds()[s][p]].imPol) {
                stagePos.push_back(p);
            }
        }
        computedOnHost[s] = runsOf(std::move(stagePos));
    }
    witnessRuns = runsOf(air.witnessColumns());
    interpreter = std::make_unique<ExpressionsGpu>(air.bin(), info, DeviceCode{args(), numbers()});

    interpolateFixed(fixedCoefs, fixedPolys);
    TimerStart(PILFFLONK_GPU_SHIFT_SUMS);
    for (const LayoutEntry &entry : info.layout) {
        if (entry.stage != info.qStage()) {
            key.addShiftSum(entry.degree, key.arena() + layout.work);
        }
    }
    // And those of the opening's W and W' (OpeningGpu), of at most the largest degree.
    const ShplonkBounds opening = shplonkBounds(air);
    key.addShiftSum(opening.wMsm, key.arena() + layout.work);
    key.addShiftSum(opening.wpMsm, key.arena() + layout.work);
    TimerStopAndLog(PILFFLONK_GPU_SHIFT_SUMS);
    commitFixed();
}

GpuAirKey::~GpuAirKey() = default;

void GpuAirKey::interpolateFixed(FrElement *fixedCoefs, std::vector<std::unique_ptr<Poly>> &fixedPolys) {
    const uint64_t nConstants = air.info().nConstants, N = air.n();
    if (nConstants == 0) {
        return;
    }
    TimerStart(PILFFLONK_FIXED_INTT);
    uint8_t *coefs = resident.data() + fixedOffset;
    Staging &staging = key.staging();
    staging.toDevice(coefs, air.fixedEvaluations(0), nConstants * N * sizeof(FrElement));
    for (uint64_t c = 0; c < nConstants; ++c) {
        transformOnDevice(coefs + c * N * sizeof(FrElement), air.info().nBits, true);
    }
    uint64_t *counts = reinterpret_cast<uint64_t *>(key.arena() + layout.counts);
    pilfflonk_gpu_count_coefficients(counts, coefs, offsetTable() + fixedColumnsStart, nConstants, N);
    std::vector<uint64_t> found(nConstants);
    staging.toHost(found.data(), counts, nConstants * sizeof(uint64_t));
    staging.toHost(fixedCoefs, coefs, nConstants * N * sizeof(FrElement));
    for (uint64_t c = 0; c < nConstants; ++c) {
        fixedPolys.push_back(mirrorPolynomial(fixedCoefs + c * N, N, found[c],
                                              air.name() + ": the fixed column " + air.info().constPolsMap[c].name));
    }
    TimerStopAndLog(PILFFLONK_FIXED_INTT);
}

void GpuAirKey::commitFixed() {
    TimerStart(PILFFLONK_FIXED_COMMITMENTS);
    for (uint64_t f = 0; f < air.nFixedF(); ++f) {
        const LayoutEntry &entry = air.info().layout[f];
        fixedPoints.push_back(key.commit(fixedCoefficients(), offsets(f), entry.k, air.n(), entry.degree,
                                         key.arena() + layout.work));
    }
    TimerStopAndLog(PILFFLONK_FIXED_COMMITMENTS);
}

const uint64_t *GpuAirKey::offsetTable() const {
    return reinterpret_cast<const uint64_t *>(resident.data() + offsetsOffset);
}

const FrElement *GpuAirKey::fixedCoefficients() const {
    return reinterpret_cast<const FrElement *>(resident.data() + fixedOffset);
}

const uint8_t *GpuAirKey::ops() const { return resident.data() + opsOffset; }

const uint32_t *GpuAirKey::args() const { return reinterpret_cast<const uint32_t *>(resident.data() + argsOffset); }

const FrElement *GpuAirKey::numbers() const {
    return reinterpret_cast<const FrElement *>(resident.data() + numbersOffset);
}

const uint64_t *GpuAirKey::witnessPositions() const {
    return reinterpret_cast<const uint64_t *>(resident.data() + positionsOffset);
}

const uint64_t *GpuAirKey::offsets(uint64_t f) const {
    if (tableStart.at(f) == ArenaLayout::NONE) {
        throw std::invalid_argument("GpuAirKey::offsets: f" + std::to_string(f) + " holds Q, which is committed on "
                                    "the host");
    }
    return offsetTable() + tableStart[f];
}

} // namespace PilFflonk
