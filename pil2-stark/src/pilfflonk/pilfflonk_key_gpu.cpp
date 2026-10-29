// The device side of a ProvingKey on the GPU (pilfflonk_key_gpu.hpp): compiled into the GPU library
// only (the Makefile's %_gpu.cpp rule), with g++. It calls the kernels of pilfflonk_kernels.cu and
// the PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu) through their C linkage.
#include "pilfflonk_key_gpu.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

#include "pilfflonk_error.hpp"
#include "pilfflonk_expressions_gpu.hpp"
#include "pilfflonk_hints_gpu.hpp"
#include "pilfflonk_kernels.hpp"
#include "pilfflonk_lde_gpu.hpp"
#include "pilfflonk_opening_gpu.hpp"
#include "pilfflonk_proving_key.hpp"
#include "pilfflonk_wrap_exec.hpp"
#include "thread_utils.hpp"
#include "timer.hpp"

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

// Each half of the Staging's pinned buffer.
constexpr uint64_t STAGING_HALF_BYTES = uint64_t(32) << 20;
// sppark's cudaMallocAsync goes through the device's default pool, which grows in steps of 32 MiB.
constexpr uint64_t POOL_GRANULE_BYTES = uint64_t(32) << 20;
// sppark's MSM: the bits of a BN128 scalar, and a bucket of its xyzz_t<fp_t>::mem_t.
constexpr uint64_t SCALAR_BITS = 254;
constexpr uint64_t BUCKET_BYTES = 4 * sizeof(FrElement);

uint64_t aligned(uint64_t bytes) { return (bytes + 255) & ~uint64_t(255); }

// The host side of a staged copy, to or from the pinned buffer: by every thread, but below
// PARALLEL_COPY_BYTES, where one thread copies sooner than an OpenMP team forks (a count or a scalar
// is 8 to 32 bytes).
constexpr uint64_t PARALLEL_COPY_BYTES = uint64_t(1) << 20;

void hostCopy(void *dst, const void *src, uint64_t bytes) {
    if (bytes < PARALLEL_COPY_BYTES) {
        std::memcpy(dst, src, bytes);
    } else {
        ThreadUtils::parcpy(dst, src, bytes, omp_get_max_threads());
    }
}

uint64_t log2Floor(uint64_t x) {
    uint64_t bits = 0;
    while (x >>= 1) {
        ++bits;
    }
    return bits;
}

// The blocks h^(256·b) of the shift's tables of an SRS of n powers: powers up to h^n, the MSMs being
// of n scalars at most.
uint64_t shiftBlocks(uint64_t n) { return (n >> 8) + 1; }

// Regions carved one after another from offset `start` (256-byte aligned), each 256-byte aligned.
class Carver {
public:
    explicit Carver(uint64_t start = 0) : end(start) {}
    uint64_t take(uint64_t bytes) {
        const uint64_t start = end;
        end = aligned(end + bytes);
        return start;
    }
    uint64_t size() const { return end; }

private:
    uint64_t end;
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
    uint64_t fixed, args, numbers, positions, offsets;
    uint64_t tableLength; // entries of the offsets' table: the polynomials of every f, then one per
                          // fixed column, then one per piece of Q
    uint64_t bytes;
};

ResidentLayout residentLayout(const AirKey &air) {
    const PilfflonkInfo &info = air.info();
    const ParserArgs &code = air.bin().expressionsBinArgsExpressions;
    ResidentLayout r{};
    r.tableLength = info.nConstants + air.nQPieces();
    for (const LayoutEntry &f : info.layout) {
        r.tableLength += f.k;
    }
    Carver carver;
    r.fixed = carver.take(info.nConstants * air.n() * sizeof(FrElement));
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

namespace {
thread_local DevicePool *activePool = nullptr;
constexpr uint64_t POOL_ALIGN = 256;
} // namespace

PoolScope::PoolScope(DevicePool *pool) : previous(std::exchange(activePool, pool)) {}

PoolScope::~PoolScope() { activePool = previous; }

DeviceBuffer::DeviceBuffer(uint64_t _bytes) : bytes(_bytes) {
    if (bytes > 0 && activePool != nullptr) {
        const uint64_t at = (activePool->bottom + POOL_ALIGN - 1) / POOL_ALIGN * POOL_ALIGN;
        if (at > activePool->top || activePool->top - at < bytes) {
            throw std::invalid_argument("DeviceBuffer: the key's device buffer has " +
                                        std::to_string(activePool->top - std::min(at, activePool->top)) +
                                        " bytes left, and it needs " + std::to_string(bytes));
        }
        memory = activePool->base + at;
        activePool->bottom = at + bytes;
        view = true;
    } else if (bytes > 0) {
        const DeviceScope device;
        void *allocated = nullptr;
        gpu_plonk_cuda_malloc(&allocated, bytes);
        memory = static_cast<uint8_t *>(allocated);
    }
}

DeviceBuffer DeviceBuffer::arenaOf(DevicePool &pool, uint64_t bytes) {
    const uint64_t at = bytes > pool.size ? 0 : (pool.size - bytes) / POOL_ALIGN * POOL_ALIGN;
    if (bytes > pool.size || at < pool.bottom) {
        throw std::invalid_argument("DeviceBuffer: the key's device buffer of " + std::to_string(pool.size) +
                                    " bytes cannot hold an arena of " + std::to_string(bytes) + " beside the " +
                                    std::to_string(pool.bottom) + " the key holds");
    }
    pool.top = std::min(pool.top, at);
    DeviceBuffer view;
    view.memory = pool.base + at;
    view.bytes = pool.size - at;
    view.view = true;
    return view;
}

DeviceBuffer::~DeviceBuffer() { release(); }

void DeviceBuffer::release() {
    if (memory != nullptr && !view) {
        const DeviceScope device;
        gpu_plonk_cuda_free(memory);
    }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : memory(std::exchange(other.memory, nullptr)), bytes(std::exchange(other.bytes, 0)),
      view(std::exchange(other.view, false)) {}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
        release();
        memory = std::exchange(other.memory, nullptr);
        bytes = std::exchange(other.bytes, 0);
        view = std::exchange(other.view, false);
    }
    return *this;
}

Staging::Staging(uint64_t halfBytes, CopyVolume &volume) : half(halfBytes), copies(volume) {
    const DeviceScope device;
    gpu_plonk_cuda_malloc_pinned_buffer(&pinned, 2 * half);
    stream = gpu_plonk_create_cuda_stream_nonblocking();
    ready = pilfflonk_gpu_event_create();
    done[0] = pilfflonk_gpu_event_create();
    done[1] = pilfflonk_gpu_event_create();
}

Staging::~Staging() {
    const DeviceScope device;
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
    toDeviceFrom(dst, bytes, [src](uint8_t *half, uint64_t at, uint64_t n) {
        hostCopy(half, static_cast<const uint8_t *>(src) + at, n);
    });
}

void Staging::toDeviceFrom(void *dst, uint64_t bytes, const Fill &fill) {
    if (bytes == 0) {
        return;
    }
    afterDefaultStream();
    uint8_t *halves[2] = {static_cast<uint8_t *>(pinned), static_cast<uint8_t *>(pinned) + half};
    uint64_t chunk = 0;
    try {
        for (uint64_t at = 0; at < bytes; at += half, ++chunk) {
            const uint64_t n = std::min(half, bytes - at);
            const uint64_t h = chunk & 1;
            // The half's previous copy is done before it is written again.
            pilfflonk_gpu_event_sync(done[h]);
            fill(halves[h], at, n);
            gpu_plonk_memcpy_h2d_async(static_cast<uint8_t *>(dst) + at, halves[h], n, stream);
            pilfflonk_gpu_event_record(done[h], stream);
        }
    } catch (...) {
        // The queued copies are done before the caller unwinds and frees `dst`.
        gpu_plonk_sync_cuda_stream(stream);
        throw;
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
        hostCopy(static_cast<uint8_t *>(dst) + chunk * half, halves[chunk & 1], length(chunk));
    }
    copies.addToHost(bytes);
}

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
    a.hintBytes = stageScratchBytes(air);
    a.stageBytes = std::max(a.stageBytes, aligned(a.work + a.hintBytes));

    // Q's phase, over the stages' (InstanceGpu::computeQ).
    const AirDegrees &d = air.degrees();
    const uint64_t nPieces = d.qPieceCoefficients.size(), NExt = air.lde().extendedSize();
    Carver q(polysEnd);
    a.qCounts = q.take((1 + nPieces) * sizeof(uint64_t));
    a.qFactors = q.take(2 * (nPieces - 1) * sizeof(FrElement));
    a.q = q.take(NExt * sizeof(FrElement));
    a.qTables = q.take(ldeTableElements(air.lde()) * sizeof(FrElement));
    a.qPart = q.take(qPartLayout(air, air.info().nBits).bytes);

    // Q's pieces (InstanceGpu::commitQ): over Q's values if it is not split, after them if it is;
    // then their commitments' work, and the opening's workspace over it.
    a.qPieceElements = *std::max_element(d.qPieceCoefficients.begin(), d.qPieceCoefficients.end());
    a.qPieces = nPieces == 1 ? a.q : aligned(a.q + NExt * sizeof(FrElement));
    a.shplonk = aligned(a.qPieces + nPieces * a.qPieceElements * sizeof(FrElement));
    uint64_t qWork = 0;
    for (const LayoutEntry &f : info.layout) {
        if (f.stage == info.qStage()) {
            qWork = std::max<uint64_t>(qWork, f.degree);
        }
    }
    const uint64_t opening =
        a.shplonk + std::max(qWork * sizeof(FrElement), shplonkWorkspaceBytes(shplonkBounds(air)));
    a.bytes = std::max({a.stageBytes, q.size(), aligned(opening)});
    return a;
}

QPartLayout qPartLayout(const AirKey &air, uint64_t partBits) {
    Carver carver;
    QPartLayout l;
    l.zhInv = carver.take(ExpressionsDomainGpu::cosetPartBytes(air.info().nBits, partBits));
    l.columns = carver.take(air.qReads().size() * (uint64_t(1) << partBits) * sizeof(FrElement));
    l.bytes = carver.size();
    return l;
}

uint64_t qPhaseBytes(const AirKey &air, const ArenaLayout &layout, uint64_t partBits) {
    return layout.qPart + qPartLayout(air, partBits).bytes;
}

uint64_t componentOffset(const AirKey &air, const ArenaLayout &layout, uint64_t f, uint64_t j) {
    const PilfflonkInfo &info = air.info();
    const LayoutEntry &entry = info.layout[f];
    if (entry.stage == 0) {
        return entry.pols[j].id * air.n();
    }
    if (entry.stage == info.qStage()) {
        return info.cmPolsMap[entry.pols[j].id].stagePos * layout.qPieceElements;
    }
    return layout.slot[f] + j * (air.n() + air.blindLength(f));
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
    const uint64_t temp = stride * 2 * 2 * sizeof(uint32_t) + nwins * stride * sizeof(uint32_t);
    // Each of the two allocations as the pool reserves it.
    const auto pooled = [](uint64_t bytes) {
        return (bytes + POOL_GRANULE_BYTES - 1) / POOL_GRANULE_BYTES * POOL_GRANULE_BYTES;
    };
    return pooled(blob) + pooled(temp);
}

namespace {
// The fewest points an MSM is split into parts of: below, sppark's MSMs slow down by a third or more
// (2^23 took 45% longer than one MSM of 2^25 on the blake3 wrap, 2^24 8%).
constexpr uint64_t MSM_PART_BITS_MIN = 22;
} // namespace

GpuBudget gpuBudget(const AirKey &air, uint32_t multiprocessors) {
    GpuBudget budget;
    budget.resident = residentLayout(air).bytes + ExpressionsGpu::deviceBytesOf(air.bin(), air.info(), multiprocessors);
    budget.arena = arenaLayout(air).bytes;
    // What sppark's MSM allocates for the largest of a proof's MSMs (GpuKey::commit).
    for (uint64_t n : air.msmLengths()) {
        budget.transient = std::max(budget.transient, spparkMsmBytes(n, multiprocessors));
        budget.longestMsm = std::max(budget.longestMsm, n);
    }
    budget.leastTransient =
        spparkMsmBytes(std::min(budget.longestMsm, uint64_t(1) << MSM_PART_BITS_MIN), multiprocessors);
    budget.loading = loadScratchBytes(air);
    return budget;
}

uint64_t loadScratchBytes(const AirKey &air) { return aligned(air.info().nConstants * sizeof(uint64_t)); }

DeviceBytes deviceBytesOf(uint64_t nG1, const std::vector<const AirKey *> &airs, uint32_t multiprocessors) {
    // AIR after AIR, as a key on a given arena checks each before it loads (GpuKey::reserve): what it
    // holds, and the AIR's resident bytes, its loading's scratch and the least a proof allocates
    // besides (its MSMs in the smallest parts, which reserve splits them into if no larger fits), of
    // the AIRs so far.
    DeviceBytes bytes;
    uint64_t held = GpuKey::powersAndTablesBytes(nG1), transient = 0;
    bytes.beside = held;
    for (const AirKey *air : airs) {
        const GpuBudget budget = gpuBudget(*air, multiprocessors);
        bytes.arena = std::max(bytes.arena, budget.arena);
        transient = std::max(transient, budget.leastTransient);
        bytes.beside = std::max(bytes.beside, held + budget.resident + budget.loading + transient);
        held += budget.resident;
    }
    return bytes;
}

uint32_t gpuMultiprocessors() {
    const DeviceScope device;
    return pilfflonk_gpu_multiprocessors();
}

uint64_t gpuFreeBytes() {
    const DeviceScope device;
    uint64_t free = 0, total = 0;
    pilfflonk_gpu_memory(&free, &total);
    return free;
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

uint64_t GpuKey::powersAndTablesBytes(uint64_t nG1) {
    return nG1 * sizeof(G1PointAffine) + (shiftBlocks(nG1) + 256) * sizeof(FrElement) + 2 * sizeof(uint64_t);
}

GpuKey::GpuKey(const Srs &srs, GpuKeyOptions _options, const SrsPowersFile *powersFile) : options(_options) {
    if (!Gpu::available()) {
        throw std::invalid_argument("GpuKey: no GPU: CUDA sees no device of compute capability 7.0 or above (or no "
                                    "driver)");
    }
    const DeviceScope device;
    if (options.exclusive) {
        pool = DevicePool{static_cast<uint8_t *>(options.arena), options.arenaBytes, 0, options.arenaBytes};
    }
    const PoolScope pooled(devicePool());
    smCount = pilfflonk_gpu_multiprocessors();
    const uint64_t n = srs.nG1();
    const uint64_t nBlocks = shiftBlocks(n);
    const uint64_t powersBytes = n * sizeof(G1PointAffine);
    const uint64_t bytes = powersAndTablesBytes(n);
    requireDeviceMemory("the SRS's " + std::to_string(n) + " powers [τ^i]₁ and the tables of the MSMs' shift", bytes,
                        available());
    transfers = std::make_unique<Staging>(STAGING_HALF_BYTES, volume);
    // Through the pinned halves of the Staging, as the PLONK GPU prover's d_ptau from registered memory.
    if (options.exclusive) {
        powersMemory = DeviceBuffer(powersBytes);
    }
    // From the file, or from the Srs's own copy of them.
    const Staging::Fill read = [&](uint8_t *half, uint64_t at, uint64_t length) {
        if (powersFile != nullptr) {
            powersFile->read(half, at, length);
        } else {
            hostCopy(half, reinterpret_cast<const uint8_t *>(&srs.g1(0)) + at, length);
        }
    };
    powers = std::make_unique<Gpu>(
        n, [&](void *dst, uint64_t size) { transfers->toDeviceFrom(dst, size, read); }, powersMemory.data());
    srsSerial = srs.serial();
    tables = DeviceBuffer(bytes - powersBytes);
    hBlocks = tables.data();
    hPowers = tables.data() + nBlocks * sizeof(FrElement);
    zeroOffset = reinterpret_cast<uint64_t *>(tables.data() + (nBlocks + 256) * sizeof(FrElement));
    count = zeroOffset + 1;
    gpu_plonk_precompute_omega_tables_async(tables.data(), tables.data() + nBlocks * sizeof(FrElement),
                                            &msmShiftRatio(), 256, static_cast<uint32_t>(nBlocks), nullptr);
    pilfflonk_gpu_memset_zero(zeroOffset, sizeof(uint64_t));
    held = bytes;
}

GpuKey::~GpuKey() = default;

void GpuKey::snapshot() const {
    if (!options.exclusive || !options.restorable) {
        return;
    }
    // Not under a proof of the key: the Staging is one thread's at a time.
    std::unique_lock<std::mutex> guard(leaseLock);
    leaseFree.wait(guard, [this] { return !leased; });
    const DeviceScope device;
    gpu_plonk_cuda_device_sync();
    image.reset();
    imageBytes = pool.bottom;
    TimerStart(PILFFLONK_GPU_SNAPSHOT);
    // Not on huge pages: faulted in after a proof, they compact memory synchronously (THP's defrag),
    // which took 27-58 s for 4.3 GB. The staging's copy faults the 4 KB pages in on every thread.
    void *bytes = std::malloc(imageBytes);
    if (bytes == nullptr) {
        throw std::bad_alloc();
    }
    image.reset(static_cast<uint8_t *>(bytes));
    transfers->toHost(image.get(), pool.base, imageBytes);
    TimerStopAndLog(PILFFLONK_GPU_SNAPSHOT);
    zklog.trace("pilfflonk: a copy of the key's " + std::to_string(imageBytes) + " bytes in its buffer");
}

void GpuKey::ImageFree::operator()(uint8_t *bytes) const { std::free(bytes); }

void GpuKey::restore() const {
    if (image == nullptr) {
        return;
    }
    // Not under a proof of the key: it would read what this overwrites.
    std::unique_lock<std::mutex> guard(leaseLock);
    if (leased && holder == std::this_thread::get_id()) {
        throw std::logic_error("GpuKey::restore: this thread holds a proof of the key, which would wait for itself");
    }
    leaseFree.wait(guard, [this] { return !leased; });
    const DeviceScope device;
    TimerStart(PILFFLONK_GPU_RESTORE);
    transfers->toDevice(pool.base, image.get(), imageBytes);
    gpu_plonk_cuda_device_sync();
    TimerStopAndLog(PILFFLONK_GPU_RESTORE);
}

uint64_t GpuKey::available() const {
    if (options.exclusive) {
        return pool.top - std::min(pool.bottom, pool.top);
    }
    uint64_t free = gpuFreeBytes();
    if (options.memoryLimit != 0) {
        free = std::min(free, options.memoryLimit > held ? options.memoryLimit - held : 0);
    }
    return free;
}

uint64_t GpuKey::msmPartsFitting(const std::string &air, const GpuBudget &budget, uint64_t free) const {
    if (budget.transient <= free || budget.longestMsm == 0) {
        return 0;
    }
    for (uint64_t bits = 63 - static_cast<uint64_t>(__builtin_clzll(budget.longestMsm)); bits >= MSM_PART_BITS_MIN;
         --bits) {
        const uint64_t points = uint64_t(1) << bits;
        if (points < budget.longestMsm && spparkMsmBytes(points, smCount) <= free) {
            zklog.info("pilfflonk: " + air + "'s MSMs in parts of 2^" + std::to_string(bits) + " points: " +
                       std::to_string(spparkMsmBytes(points, smCount)) + " bytes beside the buffer, of the " +
                       std::to_string(free) + " free (the longest's whole " + std::to_string(budget.transient) +
                       ")");
            return points;
        }
    }
    return 0;
}

uint64_t GpuKey::msmPart(uint64_t n) const {
    // The previous MSM's scratch is back in the device's pool only after a synchronisation.
    {
        const DeviceScope device;
        gpu_plonk_cuda_device_sync();
    }
    // What reserve saw free may since have gone to lazily loaded kernels or CUDA graphs.
    const uint64_t free = options.exclusive ? gpuFreeBytes() : available();
    const uint64_t most = msmMostPoints == 0 ? n : std::min(n, msmMostPoints);
    uint64_t part = most;
    while (part > (uint64_t(1) << MSM_PART_BITS_MIN) && spparkMsmBytes(part, smCount) > free) {
        part = uint64_t(1) << (63 - __builtin_clzll(part - 1));
    }
    if (part < most) {
        zklog.warning("pilfflonk: an MSM of " + std::to_string(n) + " points in parts of " + std::to_string(part) +
                      " instead of " + std::to_string(most) + ": " + std::to_string(free) + " bytes free");
    }
    return part;
}

void GpuKey::reserve(const std::string &air, const GpuBudget &budget) {
    if (arenaGiven() && budget.arena > options.arenaBytes) {
        throw std::invalid_argument(air + ": a proof on the GPU needs an arena of " + std::to_string(budget.arena) +
                                    " bytes, and the one given has " + std::to_string(options.arenaBytes) +
                                    " (pilfflonk/docs/performance.md#selection-memory-and-errors)");
    }
    const uint64_t growth = !arenaGiven() && budget.arena > arenaBytes ? budget.arena - arenaBytes : 0;
    // With an arena given, the loading's scratch is beside it while the AIR loads.
    const uint64_t loading = arenaGiven() ? budget.loading : 0;
    // A proof's MSMs in parts where the scratch of the longest does not fit beside what is held.
    const uint64_t others = options.exclusive ? 0 : budget.resident + growth + loading;
    const uint64_t room = options.exclusive ? gpuFreeBytes() : available();
    const uint64_t parts = msmPartsFitting(air, budget, room > others ? room - others : 0);
    if (parts != 0) {
        msmMostPoints = msmMostPoints == 0 ? parts : std::min(msmMostPoints, parts);
    }
    transient = std::max(transient, parts != 0 ? spparkMsmBytes(parts, smCount) : budget.transient);
    if (options.exclusive) {
        // In the buffer: the AIR's own and the arena's growth; beside it, a proof's transient.
        requireDeviceMemory(air + " (its fixed columns, bytecode and tables, " + std::to_string(budget.resident) +
                                " bytes, and the arena's growth to its proofs', " + std::to_string(growth) +
                                ", in the key's device buffer)",
                            budget.resident + growth, available());
        requireDeviceMemory(air + " (what a proof allocates besides, " + std::to_string(transient) + ")", transient,
                            gpuFreeBytes());
        if (growth > 0) {
            owned = DeviceBuffer::arenaOf(pool, budget.arena);
            arenaBytes = budget.arena;
            held += growth;
        }
        held += budget.resident;
        return;
    }
    requireDeviceMemory(air + " (its fixed columns, bytecode and tables, " + std::to_string(budget.resident) +
                            " bytes; the arena's growth to its proofs', " + std::to_string(growth) +
                            (arenaGiven() ? "; the scratch of its loading, " + std::to_string(loading) : "") +
                            "; and what a proof allocates besides, " + std::to_string(transient) + ")",
                        budget.resident + growth + loading + transient, available());
    if (growth > 0) {
        owned = DeviceBuffer();
        owned = DeviceBuffer(budget.arena);
        arenaBytes = budget.arena;
        held += growth;
    }
    held += budget.resident;
}

void GpuKey::copyToHost(void *dst, const void *src, uint64_t bytes) const {
    const DeviceScope device;
    gpu_plonk_memcpy_d2h(dst, src, bytes);
    volume.addToHost(bytes);
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
    Engine &E = Engine::engine;
    // In parts where the device does not hold the whole's scratch (msmPart). Each part's point at
    // infinity is sppark's failure unless its own scalars are all zero.
    const uint64_t most = msmPart(n);
    G1Point shifted = E.g1.zero();
    for (uint64_t at = 0; at < n; at += most) {
        const uint64_t length = std::min(most, n - at);
        void *scalars = static_cast<FrElement *>(work) + at;
        G1Point part = msmOnDevice(static_cast<const G1PointAffine *>(powers->devicePowers()) + at, scalars, length);
        if (E.g1.isZero(part) && !allZero(scalars, length)) {
            throw std::runtime_error("GpuKey::commit: the GPU's MSM of " + std::to_string(length) +
                                     " points gave the point at infinity for shifted scalars that are not all zero: "
                                     "msm_bn128_gpu_dev_ptr failed (or τ is a root of the shifted polynomial)");
        }
        E.g1.add(shifted, shifted, part);
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
                                    "the GPU memory of one proof at a time (the instance of the latest call on this "
                                    "thread, or its opening's): it would wait for itself");
    }
    owner.leaseFree.wait(guard, [this] { return !owner.leased; });
    owner.leased = true;
    owner.holder = std::this_thread::get_id();
    guard.unlock();
    // A shared buffer's earlier user may still be writing it.
    if (owner.arenaGiven() || owner.options.exclusive) {
        const DeviceScope device;
        gpu_plonk_cuda_device_sync();
    }
}

GpuKey::Lease::~Lease() {
    if (owner.arenaGiven() || owner.options.exclusive) {
        const DeviceScope device;
        gpu_plonk_cuda_device_sync();
    }
    {
        const std::lock_guard<std::mutex> guard(owner.leaseLock);
        owner.leased = false;
        owner.holder = std::thread::id();
    }
    // All: a restore may be waiting beside the next Lease.
    owner.leaseFree.notify_all();
}

void GpuKey::provingHere() const {
    const std::lock_guard<std::mutex> guard(leaseLock);
    if (leased) {
        holder = std::this_thread::get_id();
    }
}

ProofCall::ProofCall(const GpuKey &key) { key.provingHere(); }

// ---------------------------------------------------------------------------------------------
// GpuAirKey
// ---------------------------------------------------------------------------------------------

GpuAirKey::GpuAirKey(GpuKey &_key, const AirKey &_air) : key(_key), air(_air), layout(arenaLayout(_air)) {
    const DeviceScope device;
    const PoolScope pooled(key.devicePool());
    const PilfflonkInfo &info = air.info();
    const uint64_t N = air.n();
    checkSrsFits(air, key.nPowers());
    const GpuBudget budget = gpuBudget(air, key.multiprocessors());
    key.reserve(air.name(), budget);

    const ResidentLayout r = residentLayout(air);
    resident = DeviceBuffer(r.bytes);
    fixedOffset = r.fixed;
    argsOffset = r.args;
    numbersOffset = r.numbers;
    positionsOffset = r.positions;
    offsetsOffset = r.offsets;

    // The offsets of the polynomials each f packs (componentOffset): of a fixed f, its columns' in
    // the fixed coefficients; of a stage's, its slots in the arena; of Q's, its pieces'. Then the
    // fixed columns', and Q's pieces' in order.
    std::vector<uint64_t> table;
    tableStart.assign(info.layout.size(), ArenaLayout::NONE);
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        tableStart[f] = table.size();
        for (uint64_t j = 0; j < info.layout[f].k; ++j) {
            table.push_back(componentOffset(air, layout, f, j));
        }
    }
    fixedColumnsStart = table.size();
    for (uint64_t c = 0; c < info.nConstants; ++c) {
        table.push_back(c * N);
    }
    piecesStart = table.size();
    for (uint64_t i = 0; i < air.nQPieces(); ++i) {
        table.push_back(i * layout.qPieceElements);
    }
    if (table.size() != r.tableLength) {
        throw std::logic_error("GpuAirKey: " + air.name() + "'s table of offsets has " + std::to_string(table.size()) +
                               " entries, and its layout " + std::to_string(r.tableLength));
    }
    const ParserArgs &code = air.bin().expressionsBinArgsExpressions;
    Staging &staging = key.staging();
    staging.toDevice(resident.data() + argsOffset, code.args.data(), code.args.size() * sizeof(uint32_t));
    staging.toDevice(resident.data() + numbersOffset, code.numbers.data(), code.numbers.size() * sizeof(FrElement));
    staging.toDevice(resident.data() + positionsOffset, air.witnessColumns().data(),
                     air.witnessColumns().size() * sizeof(uint64_t));
    staging.toDevice(resident.data() + offsetsOffset, table.data(), table.size() * sizeof(uint64_t));

    witnessRuns = runsOf(air.witnessColumns());
    interpreter = std::make_unique<ExpressionsGpu>(air.bin(), info, DeviceCode{args(), numbers()});
    // A given arena is not written while the key loads: others may be using it.
    if (key.arenaGiven()) {
        loadScratch = DeviceBuffer(budget.loading);
    }

    for (uint64_t n : air.msmLengths()) {
        if (!key.hasShiftSum(n)) {
            throw FormatError(air.name() + ": the key has no shift sum of its MSMs of " + std::to_string(n) +
                              " points: its setup is older than its .coefs and pilfflonk.shift.bin (run it again)");
        }
    }
}

GpuAirKey::~GpuAirKey() = default;

void GpuAirKey::loadFixed(const uint8_t *coefficients) {
    loadFixedFrom([coefficients](uint8_t *half, uint64_t at, uint64_t n) { hostCopy(half, coefficients + at, n); });
}

void GpuAirKey::loadFixedFrom(const Staging::Fill &fill) {
    if (fixedLoaded) {
        throw std::logic_error("GpuAirKey::loadFixed: " + air.name() + "'s fixed columns are loaded already");
    }
    const DeviceScope device;
    const uint64_t nConstants = air.info().nConstants, N = air.n();
    if (nConstants > 0) {
        TimerStart(PILFFLONK_FIXED_UPLOAD);
        uint8_t *coefs = resident.data() + fixedOffset;
        Staging &staging = key.staging();
        staging.toDeviceFrom(coefs, nConstants * N * sizeof(FrElement), fill);
        pilfflonk_gpu_to_montgomery(coefs, nConstants * N);
        // Their counts only: the coefficients stay on the device, and go to the host on demand
        // (fixedToHost).
        uint64_t *counts = loadCounts();
        pilfflonk_gpu_count_coefficients(counts, coefs, offsetTable() + fixedColumnsStart, nConstants, N);
        fixedCounts.assign(nConstants, 0);
        staging.toHost(fixedCounts.data(), counts, nConstants * sizeof(uint64_t));
        TimerStopAndLog(PILFFLONK_FIXED_UPLOAD);
    }
    gpu_plonk_cuda_device_sync();
    loadScratch = DeviceBuffer();
    fixedLoaded = true;
}

void GpuAirKey::setExec(const pilfflonk_exec_static &exec) {
    const uint64_t N = air.n(), C = air.witnessColumns().size();
    // The additions kernel counts levels in a uint8_t, which wraps past 255.
    if (execParts.size() != 0 || exec.map_rows > N || exec.map_cols > C || (exec.n_adds > 0 && exec.n_levels == 0) ||
        exec.n_levels > 255 || exec.n_wires + exec.n_adds > UINT32_MAX) {
        throw std::invalid_argument("GpuAirKey::setExec: parts of an exec that is not " + air.name() +
                                    "'s, or set already");
    }
    // The device reads them unchecked: an addition reads wires or earlier additions, the map up to
    // the zero past them.
    const uint64_t cells = exec.map_rows * exec.map_cols;
    bool outOfRange = false;
#pragma omp parallel for reduction(|| : outOfRange)
    for (uint64_t i = 0; i < exec.n_adds; ++i) {
        const uint32_t reads[2] = {exec.add_wire1[i], exec.add_wire2[i]};
        bool bad = exec.add_level[i] >= exec.n_levels;
        for (uint32_t w : reads) {
            // An addition read is of a level below: the kernel runs the levels in order.
            bad = bad || w >= exec.n_wires + i || (w >= exec.n_wires && exec.add_level[w - exec.n_wires] >= exec.add_level[i]);
        }
        outOfRange = outOfRange || bad;
    }
#pragma omp parallel for reduction(|| : outOfRange)
    for (uint64_t i = 0; i < cells; ++i) {
        outOfRange = outOfRange || exec.map[i] > exec.n_wires + exec.n_adds;
    }
    if (outOfRange) {
        throw std::invalid_argument("GpuAirKey::setExec: " + air.name() +
                                    "'s exec has an addition or a map cell past its wires");
    }
    const DeviceScope device;
    // With the key's other memory: in its buffer, and so in its snapshot, if it has one.
    const PoolScope pooled(key.devicePool());
    const uint64_t fr = sizeof(FrElement);
    auto aligned256 = [](uint64_t bytes) { return (bytes + 255) / 256 * 256; };
    const uint64_t sizes[6] = {cells * 4, exec.n_adds * 4, exec.n_adds * 4, exec.n_adds * fr, exec.n_adds * fr,
                               exec.n_adds};
    uint64_t total = 0;
    for (uint64_t b : sizes) {
        total += aligned256(b);
    }
    execParts = DeviceBuffer(total);
    uint8_t *at = execParts.data();
    const void *sources[6] = {exec.map, exec.add_wire1, exec.add_wire2, exec.add_coef1, exec.add_coef2, exec.add_level};
    const uint8_t **targets[6] = {&execOn.map, &execOn.id1, &execOn.id2, &execOn.f1, &execOn.f2, &execOn.levels};
    Staging &staging = key.staging();
    for (int i = 0; i < 6; ++i) {
        staging.toDevice(at, sources[i], sizes[i]);
        *targets[i] = at;
        at += aligned256(sizes[i]);
    }
    // As the PLONK GPU prover: the factors in Montgomery form, the witness in normal form.
    pilfflonk_gpu_to_montgomery(const_cast<uint8_t *>(execOn.f1), exec.n_adds);
    pilfflonk_gpu_to_montgomery(const_cast<uint8_t *>(execOn.f2), exec.n_adds);
    gpu_plonk_cuda_device_sync();
    execOn.nWires = exec.n_wires;
    execOn.nAdds = exec.n_adds;
    execOn.nLevels = exec.n_levels;
    execOn.mapRows = exec.map_rows;
    execOn.mapCols = exec.map_cols;
}

uint64_t *GpuAirKey::loadCounts() const {
    uint8_t *counts = key.arenaGiven() ? loadScratch.data() : key.arena() + layout.counts;
    return reinterpret_cast<uint64_t *>(counts);
}

const std::vector<G1Point> &GpuAirKey::fixedCommitments() const {
    std::call_once(fixedCommitted, [this] {
        const GpuKey::Lease lease(key, "GpuAirKey::fixedCommitments");
        const DeviceScope device;
        TimerStart(PILFFLONK_FIXED_COMMITMENTS);
        for (uint64_t f = 0; f < air.nFixedF(); ++f) {
            const LayoutEntry &entry = air.info().layout[f];
            fixedPoints.push_back(key.commit(fixedCoefficients(), offsets(f), entry.k, air.n(), entry.degree,
                                             key.arena() + layout.work));
        }
        TimerStopAndLog(PILFFLONK_FIXED_COMMITMENTS);
    });
    return fixedPoints;
}

uint64_t GpuAirKey::fixedDegree(uint64_t c) const {
    const uint64_t n = fixedCount(c);
    return n == 0 ? 0 : n - 1;
}

void GpuAirKey::fixedToHost(FrElement *out) const {
    key.copyToHost(out, fixedCoefficients(), air.info().nConstants * air.n() * sizeof(FrElement));
}

const uint64_t *GpuAirKey::offsetTable() const {
    return reinterpret_cast<const uint64_t *>(resident.data() + offsetsOffset);
}

const FrElement *GpuAirKey::fixedCoefficients() const {
    return reinterpret_cast<const FrElement *>(resident.data() + fixedOffset);
}

const uint32_t *GpuAirKey::args() const { return reinterpret_cast<const uint32_t *>(resident.data() + argsOffset); }

const FrElement *GpuAirKey::numbers() const {
    return reinterpret_cast<const FrElement *>(resident.data() + numbersOffset);
}

const uint64_t *GpuAirKey::witnessPositions() const {
    return reinterpret_cast<const uint64_t *>(resident.data() + positionsOffset);
}

const uint64_t *GpuAirKey::offsets(uint64_t f) const { return offsetTable() + tableStart.at(f); }

const uint64_t *GpuAirKey::qPieceOffsets() const { return offsetTable() + piecesStart; }

} // namespace PilFflonk
