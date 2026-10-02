#ifndef PILFFLONK_KEY_GPU_HPP
#define PILFFLONK_KEY_GPU_HPP

// The device side of a ProvingKey on the GPU (pilfflonk/docs/performance.md#gpu): what stays on the
// device while the key lives, its fixed commitments, and the memory its proofs use there. Only a
// library built with the GPU (__USE_CUDA__) has it, in pilfflonk_key_gpu.cpp; the classes that
// use it hold it under __USE_CUDA__.
//
// The device is device 0, as for Gpu: every function here that a caller reaches makes it the calling
// thread's current device while it runs (DeviceScope; ProofCall for a proof's), and so does the
// release of what holds device memory, a stream or an event (DeviceBuffer, Staging, Gpu), wherever
// it is released. Kernels run on the legacy default stream, copies to and from
// pageable memory on one non-blocking copy stream (Staging) ordered with it by events, and the
// device is synchronised before every call of sppark's MSM and NTT, which run on streams of their
// own (msmOnDevice, transformOnDevice). A CUDA failure aborts the process, as in Gpu.

#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "pilfflonk_commit.hpp"
#include "pilfflonk_gpu.hpp"
#include "pilfflonk_srs.hpp"

namespace PilFflonk {

class AirKey;         // pilfflonk_proving_key.hpp
class ExpressionsGpu; // pilfflonk_expressions_gpu.hpp
class ProofCall;      // below

// Device memory on the GPU's device, freed with it.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(uint64_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(DeviceBuffer &&other) noexcept;
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    uint8_t *data() const { return memory; }
    uint64_t size() const { return bytes; }

private:
    // Frees the memory, if any.
    void release();

    uint8_t *memory = nullptr;
    uint64_t bytes = 0;
};

// Copies between pageable host memory and the device through two pinned halves, on a non-blocking
// copy stream of its own, as the PLONK GPU prover's double-buffered copies
// (gpu_plonk_start_cpu_to_gpu_transfer): the host copies one half (on every OpenMP thread) while
// the other half's copy to or from the device runs. Each copy waits for the work the default stream
// has before it, and the default stream's work after an upload waits for the upload. Every copy is
// counted in `volume`. Not safe from several threads at once: the holder of a GpuKey's Lease, or
// the key's loading, uses it.
class Staging {
public:
    Staging(uint64_t halfBytes, CopyVolume &volume);
    ~Staging();
    Staging(const Staging &) = delete;
    Staging &operator=(const Staging &) = delete;

    // `bytes` bytes from host `src` to device `dst`. Returns once src may change.
    void toDevice(void *dst, const void *src, uint64_t bytes);
    // `bytes` bytes from device `src` to host `dst`. Returns once they are there.
    void toHost(void *dst, const void *src, uint64_t bytes);

private:
    // Makes the copy stream wait for the work the default stream has so far.
    void afterDefaultStream();

    uint64_t half;
    CopyVolume &copies;
    void *pinned = nullptr;
    void *stream = nullptr;
    void *ready = nullptr;     // the default stream's work so far
    void *done[2] = {};        // each half's last copy
};

// Where a proof of an AIR keeps its data in a GpuKey's arena: byte offsets from its start, each a
// multiple of 256, and counts of 32-byte elements. The phases of a proof (Instance) reuse the bytes
// of the ones before them, but for the committed polynomials, which last the whole proof, and Q's
// pieces, which last from Q's commitment to its end:
// - the committed polynomials, at `polys`: those of the f of stages 1 … nStages, in the order of the
//   layout and, within an f, of its columns, p_j of f at element slot[f] + j·(N + b_f), each with
//   room for its blinding (b_f = AirKey::blindLength(f));
// - the stages': the columns of each stage s on H, column p at evaluations[s] + p·N elements; the
//   work buffer, which holds the stage-1 witness as the Instance is given it, and then each f packed
//   and shifted for its MSM (GpuKey::commit); and a stage's blinding factors and coefficient counts;
//   and from the work buffer on, at `hints`, the scratch of the hints and im pols of a stage
//   (StageScratch, pilfflonk_hints_gpu.hpp), which are computed before the stage's commit uses that
//   memory (and, in stage 1, after the Instance has transposed the witness);
// - Q's (InstanceGpu::computeQ), after the committed polynomials: at `qCounts`, the 64-bit counts of
//   the coefficients of Q and of its pieces (1 + nQPieces); at `qFactors`, the two blinding factors
//   of each boundary between pieces; at `q`, Q's N' values on the coset, by point, and then its
//   coefficients; at `qTables`, the tables of the powers of a shift (ldeTableElements); and at
//   `qPart`, Zi of a part of the coset and every column Q reads on it (qPartLayout), the default
//   parts of N points: Q's phase ends at qPhaseBytes(air, layout, nBits), and larger parts need
//   qPhaseBytes(air, layout, partBits) bytes of the arena;
// - Q's pieces (InstanceGpu::commitQ), at `qPieces`, piece i at i·qPieceElements elements, the most
//   of AirDegrees::qPieceCoefficients, and zero above its own: Q itself, over its coefficients at
//   `q`, if it is not split, and after them if it is; then, at `shplonk`, the work of their
//   commitments (the largest degree of an f of Q's), and later the opening's SHPLONK workspace
//   (OpeningGpu, pilfflonk_opening_gpu.hpp).
struct ArenaLayout {
    static constexpr uint64_t NONE = UINT64_MAX;

    uint64_t polys = 0;
    uint64_t polyElements = 0;
    std::vector<uint64_t> slot; // by f; NONE for the fixed f and Q's
    std::vector<uint64_t> evaluations; // by stage, 1 … nStages
    uint64_t work = 0;
    uint64_t workElements = 0; // max(N·C, the largest f's degree)
    uint64_t factors = 0;
    uint64_t factorElements = 0; // the most of a stage, Σ k_f·b_f
    uint64_t counts = 0;
    uint64_t nCounts = 0; // 64-bit counts: the most polynomials of a stage, or the fixed columns
    uint64_t hints = 0;     // = work
    uint64_t hintBytes = 0; // the most of a stage (stageScratchBytes)
    uint64_t stageBytes = 0; // the end of the stages' phase
    uint64_t qCounts = 0;
    uint64_t qFactors = 0;
    uint64_t q = 0;
    uint64_t qTables = 0;
    uint64_t qPart = 0;
    uint64_t qPieces = 0;
    uint64_t qPieceElements = 0;
    uint64_t shplonk = 0;
    uint64_t bytes = 0; // a proof's arena: its largest phase
};

ArenaLayout arenaLayout(const AirKey &air);

// Where a part of 2^partBits points of the coset keeps its data at ArenaLayout::qPart: Zi of the
// part (ExpressionsDomainGpu::cosetPart's workspace) at `zerofiers`, and column r of AirKey::qReads
// at `columns` + r·2^partBits elements; byte offsets from qPart, and `bytes` in all.
struct QPartLayout {
    uint64_t zerofiers = 0;
    uint64_t columns = 0;
    uint64_t bytes = 0;
};

QPartLayout qPartLayout(const AirKey &air, uint64_t partBits);

// The bytes of the arena a proof of `air` (whose arena `layout` describes) needs for Q's phase in
// parts of 2^partBits points (nBits <= partBits <= nBitsExt), from the arena's start: layout.bytes
// covers it for the default parts of N points, and a larger part needs as many bytes of the key's
// arena (InstanceGpu::requireQParts).
uint64_t qPhaseBytes(const AirKey &air, const ArenaLayout &layout, uint64_t partBits);

// Where p_j of f is on the device, in elements: from GpuAirKey::fixedCoefficients() for a fixed f,
// from the arena's committed polynomials (ArenaLayout::polys) for an f of a committed stage, and
// from the arena's pieces of Q (ArenaLayout::qPieces) for an f of Q's, whose p_j is a piece.
uint64_t componentOffset(const AirKey &air, const ArenaLayout &layout, uint64_t f, uint64_t j);

// The device memory a key on the GPU needs for an AIR, in bytes.
struct GpuBudget {
    uint64_t resident = 0;  // held while the key lives: its fixed columns' coefficients, bytecode, interpreter
                            // (ExpressionsGpu::deviceBytesOf) and tables
    uint64_t arena = 0;     // a proof's arena (ArenaLayout::bytes)
    uint64_t transient = 0; // what a proof allocates besides, while it runs
    uint64_t loading = 0;   // the scratch of the AIR's loading (loadScratchBytes): in the arena, or
                            // beside it while the AIR loads if the arena is given
};

// The budget of `air` on a device of `multiprocessors` SMs. `resident` counts the device memory of its
// interpreter (ExpressionsGpu::deviceBytesOf); `transient` is the most that sppark's MSM allocates
// for the largest f (msm_t of pippenger.cuh, which grows with the SMs), and a margin for the
// allocator's rounding and sppark's NTT tables.
GpuBudget gpuBudget(const AirKey &air, uint32_t multiprocessors);

// The scratch GpuAirKey uses while it loads, in bytes: the shift's scalars of its longest MSM
// (GpuKey::addShiftSums) and the fixed f packed for theirs (GpuKey::commit), and the counts of the
// fixed columns' coefficients. Within the arena's work buffer and counts (arenaLayout).
uint64_t loadScratchBytes(const AirKey &air);

// The device memory a key on the GPU of an SRS of nG1 powers and of `airs` (in the order they load)
// needs on a device of `multiprocessors` SMs, as GpuKey and its AIRs (GpuAirKey) reserve it
// (DeviceBytes): the arena of the largest proof, and beside a given arena the most the key holds and
// allocates as each AIR loads, the SRS's powers and tables (GpuKey::powersAndTablesBytes), the
// resident bytes of the AIRs so far, the loading's scratch of the AIR and the most a proof of them
// allocates besides (GpuBudget), what GpuKey::reserve checks against the free memory.
DeviceBytes deviceBytesOf(uint64_t nG1, const std::vector<const AirKey *> &airs, uint32_t multiprocessors);

// The multiprocessors of the device a GpuKey uses, device 0, and its free memory in bytes
// (cudaMemGetInfo), whatever the calling thread's current device (DeviceScope).
uint32_t gpuMultiprocessors();
uint64_t gpuFreeBytes();

// The bytes sppark's MSM allocates for itself (msm_t and its invoke, pippenger.cuh) for n points and
// scalars already on a device of `multiprocessors` SMs.
uint64_t spparkMsmBytes(uint64_t n, uint32_t multiprocessors);

// Throws std::invalid_argument, saying that `what` needs `needed` bytes of device memory and the
// device has `available` free, if needed > available.
void requireDeviceMemory(const std::string &what, uint64_t needed, uint64_t available);

// Logs at -vv, with the PILFFLONK_* timers, the bytes `volume` counted since `since`:
// "PILFFLONK_COPIES_<phase>: <n> bytes to the device, <m> to the host".
void logCopies(const CopyVolume &volume, const std::string &phase, const CopyVolume::Totals &since);

// The GPU side of a ProvingKey: the SRS's powers [τ^i]₁ on the device (its Gpu), the tables of the
// MSM's shift and its sums, the arena where a proof keeps its data (ArenaLayout), its own or one given
// to it (GpuKeyOptions::arena), and the Staging of its copies.
//
// The MSM (commit) shifts its scalars: it computes Σ (s_i + ρ_i)·[τ^i]₁ − Σ ρ_i·[τ^i]₁, with
// ρ_i = h^(i+1), h = msmShiftRatio(). sppark's Pippenger is fast when the scalars look random and slow
// when many of them repeat, as every coefficient of L1 does (in each window, every point whose scalar
// has the same digit goes to one bucket, which few threads sum: on an RTX 5090, 2^23 points take
// 0.05 s with random scalars and 26 s with equal ones); s + ρ looks random whatever s is, and the
// difference is the same point. A kernel adds ρ while it packs the scalars, from the tables
// h^(256·b) and h^t (t < 256) that gpu_plonk_precompute_omega_tables_async computes with base h.
// Every MSM of the device path is of a static length, the degree bound of its f in the layout (zero
// scalars add nothing, so the commitment is the same), and the sum Σ_{i<n} ρ_i·[τ^i]₁ of each length
// is computed once, when the AIRs that commit with it load (addShiftSums): each from the one of the
// next shorter length, with an MSM of the points between them, so that the sums of all the lengths
// take the MSM of the longest.
//
// Its AIRs (GpuAirKey) reserve their memory as they load, and each is refused, with how much it needs
// and how much is free, before anything of it goes to the device, if the device cannot hold it and a
// whole proof of it with the key's other data: a key on the GPU never runs out of device memory in the
// middle of a proof (reserve). With an arena given (GpuKeyOptions::arena) that is smaller than a
// proof's, it is refused too.
//
// One proof at a time uses the arena and the Staging: the one whose Instance holds a Lease, which
// another thread's waits for. Its other const functions are safe from several threads.
class GpuKey {
public:
    // Copies the powers [τ^i]₁ of `srs` to the device, through the pinned Staging, and builds the
    // shift's tables. Throws std::invalid_argument if there is no GPU (Gpu::available) or they do not
    // fit in its memory.
    explicit GpuKey(const Srs &srs, GpuKeyOptions options = GpuKeyOptions());
    ~GpuKey();
    GpuKey(const GpuKey &) = delete;
    GpuKey &operator=(const GpuKey &) = delete;

    // The device memory of the powers [τ^i]₁ of an SRS of nG1 powers and of the shift's tables, in
    // bytes: what the constructor holds.
    static uint64_t powersAndTablesBytes(uint64_t nG1);

    uint64_t nPowers() const { return powers->nPoints(); }
    // Whether these are the powers [τ^i]₁ of `srs`: the key was made from it (or from the Srs it was
    // moved from), whose powers it copied.
    bool holds(const Srs &srs) const { return &srs.g1(0) == hostPowers; }
    uint32_t multiprocessors() const { return smCount; }
    CopyVolume &copies() const { return volume; }
    // The bytes the key holds on the device, its arena included if it is its own.
    uint64_t deviceBytes() const { return held; }
    // The most a proof of its AIRs allocates besides, while it runs (GpuBudget::transient).
    uint64_t transientBytes() const { return transient; }

    // While the key loads, before it is shared (GpuAirKey's constructor), before anything of the AIR
    // goes to the device: room for an AIR whose budget is `budget`, and, with an arena given, its
    // loading's scratch (GpuBudget::loading, which GpuAirKey allocates for the time it loads). Its own
    // arena grows to the largest proof's. Throws std::invalid_argument, naming `air`, if the device has
    // not the memory (requireDeviceMemory), or the arena given in the options is smaller than a
    // proof's.
    void reserve(const std::string &air, const GpuBudget &budget);
    // While the key loads: Σ_{i<n} ρ_i·[τ^i]₁ for each n of `lengths` (each <= nPowers()) it has not
    // yet, for commit's MSMs of n scalars, with `work` (the longest's elements on the device) as
    // scratch: each from the sum of the next shorter length it has, if any, and an MSM of the points
    // between them (msmOnDevice of the powers from there), in increasing order. Throws
    // std::runtime_error if an MSM fails.
    void addShiftSums(std::vector<uint64_t> lengths, void *work);
    // addShiftSums of n alone.
    void addShiftSum(uint64_t n, void *work);

    // For the holder of a Lease (a given arena is written only then).
    uint8_t *arena() const { return arenaGiven() ? static_cast<uint8_t *>(options.arena) : owned.data(); }
    // Its bytes: the given arena's, or the largest proof's of the AIRs that reserved it.
    uint64_t arenaSize() const { return arenaGiven() ? options.arenaBytes : arenaBytes; }
    // Whether the arena is given (GpuKeyOptions::arena), not its own.
    bool arenaGiven() const { return options.arena != nullptr; }
    Staging &staging() const { return *transfers; }

    // `bytes` bytes from device `src` to host `dst`, counted, from any thread: a plain copy, which
    // waits for the work on the legacy default stream before it, for the copies on demand of data
    // that stays on the device while the key lives (AirKey::fixedPolynomial).
    void copyToHost(void *dst, const void *src, uint64_t bytes) const;

    // [f(τ)]₁ of f(X) = Σ_{j<k} p_j(X^k)·X^j, for k polynomials of `length` coefficients on the device,
    // p_j at base + offsets[j] elements (offsets on the device), committed with an MSM of n scalars,
    // n >= k·length and a length of addShiftSum: packed and shifted into `work` (n elements), MSM,
    // and the shift's sum subtracted. Throws std::logic_error if n has no shift sum, and
    // std::runtime_error if the MSM fails (sppark's point at infinity for scalars not all zero).
    G1Point commit(const void *base, const uint64_t *offsets, uint64_t k, uint64_t length, uint64_t n,
                   void *work) const;

    // The arena of one proof (an Instance's), held from its construction to its end; another thread's
    // waits for it. Throws std::invalid_argument, naming `function`, if this thread holds it already:
    // it would wait for itself. The thread that holds it is the one of the latest call of the proof
    // that holds it on the device (its Instance's construction, or a ProofCall of it or of its
    // opening), which may not be the one that made it: an instance may move to another thread, which
    // then holds it from its first call there. A given arena is the caller's other work's between the
    // proofs: the device is synchronised once the lease is taken, before the proof writes it, and
    // before the lease is let go, after the proof's last work on it.
    class Lease {
    public:
        Lease(const GpuKey &key, const char *function);
        ~Lease();
        Lease(const Lease &) = delete;
        Lease &operator=(const Lease &) = delete;

    private:
        const GpuKey &owner;
    };

private:
    friend class ProofCall;

    // What the device has free for the key: cudaMemGetInfo's, within options.memoryLimit.
    uint64_t available() const;
    // While the lease is held: the calling thread is the one that holds it (ProofCall).
    void provingHere() const;
    // Whether the n scalars at `scalars` on the device are all zero.
    bool allZero(const void *scalars, uint64_t n) const;

    GpuKeyOptions options;
    mutable CopyVolume volume;
    uint32_t smCount = 0;
    uint64_t held = 0;
    uint64_t transient = 0; // the most of the AIRs' GpuBudget::transient
    std::unique_ptr<Gpu> powers;
    const G1PointAffine *hostPowers = nullptr; // those of the Srs it copied
    DeviceBuffer tables;    // h^(256·b), h^t, then a 64-bit 0 (an offset) and a 64-bit count
    const void *hBlocks = nullptr;
    const void *hPowers = nullptr;
    uint64_t *zeroOffset = nullptr;
    uint64_t *count = nullptr;
    std::map<uint64_t, G1Point> shiftSums;
    DeviceBuffer owned;
    uint64_t arenaBytes = 0;
    std::unique_ptr<Staging> transfers;

    mutable std::mutex leaseLock;
    mutable std::condition_variable leaseFree;
    mutable bool leased = false;
    mutable std::thread::id holder;
};

// What every call of a proof on the device holds while it runs (InstanceGpu's, OpeningGpu's,
// computeStageColumns and stageColumnToHost, for the instance that holds `key`'s Lease): the GPU's
// device made current (DeviceScope), and the calling thread recorded as the one that holds the lease
// (GpuKey::Lease), on which a second instance of the key is refused.
class ProofCall {
public:
    explicit ProofCall(const GpuKey &key);
    ProofCall(const ProofCall &) = delete;
    ProofCall &operator=(const ProofCall &) = delete;

private:
    DeviceScope device;
};

// The device side of an AirKey on the GPU: its fixed columns' coefficients, its bytecode (the args
// and numbers of its expressions' code, which the interpreter on the device runs) and that
// interpreter, and the
// tables its kernels read, all on the device while the key lives; its fixed commitments; and where
// its proofs keep their data in the GpuKey's arena (ArenaLayout). Nothing of it is copied back to the
// host but the counts of its fixed columns' coefficients, and their coefficients when they are asked
// for (fixedToHost).
//
// It loads in two steps, so that the host reads the AIR's .const while the device works on what does
// not need it (ProvingKey::load): the constructor, and then loadFixed with the fixed columns on H.
class GpuAirKey {
public:
    // The first step of `air`'s, whose host side is derived (but for its fixed columns): reserves its
    // memory in `key` (GpuKey::reserve) before anything of it goes to the device; copies its bytecode
    // and makes its interpreter on the device; and computes the shift's sums of its MSMs' lengths
    // (GpuKey::addShiftSums). Throws FormatError if an f has more coefficients than the SRS has powers
    // (checkSrsFits), and as reserve and addShiftSums.
    GpuAirKey(GpuKey &key, const AirKey &air);
    ~GpuAirKey();
    GpuAirKey(const GpuAirKey &) = delete;
    GpuAirKey &operator=(const GpuAirKey &) = delete;

    // The second step, once: the fixed columns on H (fixedEvaluations, column c at c·N, as
    // AirKey::fixedEvaluations) to the device, interpolated there (sppark's INTT) into the
    // coefficients that stay there, the counts of their coefficients back, and the fixed commitments.
    // Throws as GpuKey::commit, and std::logic_error if called twice.
    void loadFixed(const FrElement *fixedEvaluations);

    const GpuKey &gpuKey() const { return key; }
    const AirKey &airKey() const { return air; }
    const ArenaLayout &arena() const { return layout; }

    // The commitments of the fixed f, in the order of the layout, as AirKey::fixedCommitments.
    const std::vector<G1Point> &fixedCommitments() const { return fixedPoints; }

    // The degree of fixed column c's polynomial (0 if it is zero), as the device found it.
    uint64_t fixedDegree(uint64_t c) const;
    // 1 + that degree, 0 if it is zero: pilfflonk_gpu_count_coefficients's count.
    uint64_t fixedCount(uint64_t c) const { return fixedCounts.at(c); }
    // The fixed columns' coefficients, N per column, copied to `out` (host memory) with
    // GpuKey::copyToHost: for the copies on demand of AirKey::fixedPolynomial. Safe from any thread.
    void fixedToHost(FrElement *out) const;

    // On the device: the fixed columns' coefficients (column c at c·N); the args and the numbers of
    // the expressions' code (ExpressionsBin::expressionsBinArgsExpressions), which the interpreter
    // reads (its ops it does not: an op's args say what it is); the stagePos in stage 1 of each
    // witness column (AirKey::witnessColumns); the offsets of the polynomials f packs
    // (componentOffset); and those of Q's pieces, piece i at qPieceOffsets()[i], from
    // ArenaLayout::qPieces (the first is 0).
    const FrElement *fixedCoefficients() const;
    const uint32_t *args() const;
    const FrElement *numbers() const;
    const uint64_t *witnessPositions() const;
    const uint64_t *offsets(uint64_t f) const;
    const uint64_t *qPieceOffsets() const;

    // The interpreter on the device of the expressions' code above (pilfflonk_expressions_gpu.hpp):
    // the stages' hints and im pols, and Q; for the holder of the arena, one proof at a time.
    const ExpressionsGpu &expressions() const { return *interpreter; }

    // Runs of consecutive stagePos, (first, count), in increasing order, of the witness columns in
    // stage 1.
    using ColumnRuns = std::vector<std::pair<uint64_t, uint64_t>>;
    const ColumnRuns &witnessColumns() const { return witnessRuns; }

private:
    void interpolateFixed(const FrElement *fixedEvaluations);
    void commitFixed();
    const uint64_t *offsetTable() const;
    // The scratch of the loading (loadScratchBytes): the arena's work buffer and counts, or, with an
    // arena given, which the key never writes while it loads, a buffer of its own.
    void *loadWork() const;
    uint64_t *loadCounts() const;

    GpuKey &key;
    const AirKey &air;
    ArenaLayout layout;
    DeviceBuffer resident;
    uint64_t fixedOffset = 0, argsOffset = 0, numbersOffset = 0, positionsOffset = 0, offsetsOffset = 0;
    std::vector<uint64_t> tableStart; // by f, its first entry in the offsets' table
    uint64_t fixedColumnsStart = 0;   // the entries c·N of the fixed columns, for their counts
    uint64_t piecesStart = 0;         // the entries of Q's pieces
    ColumnRuns witnessRuns;
    std::vector<uint64_t> fixedCounts; // by fixed column, once loadFixed has run
    std::vector<G1Point> fixedPoints;
    std::unique_ptr<ExpressionsGpu> interpreter;
    DeviceBuffer loadScratch; // with an arena given, until loadFixed ends
    bool fixedLoaded = false;
};

// The host copy of a polynomial of `length` coefficients just copied from the device to `coefs`,
// as Poly::fromReservedBuffer makes it (over coefs, which it does not own, its degree fixed), whose
// count of coefficients the device found (pilfflonk_gpu_count_coefficients). Throws
// std::logic_error, naming `what`, if its degree is not the device's: the copy is not what the
// device holds.
std::unique_ptr<Poly> mirrorPolynomial(FrElement *coefs, uint64_t length, uint64_t deviceCount,
                                       const std::string &what);

// Logs, when it goes, the bytes copied by `key` since it was made (logCopies), as phase `phase`;
// nothing with no key.
class CopyLog {
public:
    CopyLog(const GpuKey *key, std::string phase);
    ~CopyLog();
    CopyLog(const CopyLog &) = delete;
    CopyLog &operator=(const CopyLog &) = delete;

private:
    const GpuKey *key;
    std::string name;
    CopyVolume::Totals start;
};

} // namespace PilFflonk

#endif
