#ifndef MULTIPLICITY_CUH
#define MULTIPLICITY_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <atomic>
#include <vector>
#include <map>
#include <utility>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <string>
#include <tuple>
#include <thread>
#include "zklog.hpp"
#include "exit_process.hpp"
#include "cuda_utils.cuh"
#include "multiplicity.hpp"


// ---------------------------------------------------------------------------------------------
// Every host<->device operation below runs on a dedicated per-device NON-BLOCKING stream, never on
// the default one. The default stream is legacy-synchronising: issuing a copy or a memset on it
// while any other thread is capturing a graph fails with "operation would make the legacy stream
// depend on a capturing blocking stream", and poisons that capture. Commits, recursion and these
// transfers all run concurrently, so touching the legacy stream here is never safe.
inline cudaStream_t mulXferStream(int gpuId) {
    static std::map<int, cudaStream_t> m;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lk(mtx);
    auto it = m.find(gpuId);
    if (it != m.end()) return it->second;
    cudaStream_t s = nullptr;
    CHECKCUDAERR(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    m[gpuId] = s;
    return s;
}

inline void mulMemsetSync(int gpuId, void* dst, int val, size_t bytes) {
    cudaStream_t s = mulXferStream(gpuId);
    CHECKCUDAERR(cudaMemsetAsync(dst, val, bytes, s));
    CHECKCUDAERR(cudaStreamSynchronize(s));
}

inline void mulCopySync(int gpuId, void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    cudaStream_t s = mulXferStream(gpuId);
    CHECKCUDAERR(cudaMemcpyAsync(dst, src, bytes, kind, s));
    CHECKCUDAERR(cudaStreamSynchronize(s));
}

// Scatter completion without a device-wide wait. cudaDeviceSynchronize is also illegal while
// another thread captures, so instead every scatter records an event on the stream it was issued
// on and the fold waits on exactly those.
struct MulEventPool { std::vector<cudaEvent_t> pending, freelist; };

inline std::map<int, MulEventPool>& mulEventPools() { static std::map<int, MulEventPool> m; return m; }
inline std::mutex& mulEventMutex() { static std::mutex m; return m; }

inline void mul_note_scatter(int gpuId, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(mulEventMutex());
    MulEventPool& p = mulEventPools()[gpuId];
    cudaEvent_t e = nullptr;
    if (!p.freelist.empty()) { e = p.freelist.back(); p.freelist.pop_back(); }
    else CHECKCUDAERR(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CHECKCUDAERR(cudaEventRecord(e, stream));
    p.pending.push_back(e);
}

// Optional profiling of the scatter itself, behind ZISK_MUL_PROFILE. A start/stop event pair per
// phase, summed once at the fold: recording an event is cheap and the elapsed time is only read
// after the fold has already synchronised, so this adds no synchronisation of its own.
enum MulPhase { MUL_PHASE_KERNEL = 0, MUL_PHASE_INTERP = 1, MUL_PHASE_N = 2 };

struct MulPhaseSpan { cudaEvent_t start, stop; int phase; uint64_t air; uint64_t jobs; uint64_t rows; };

inline std::vector<MulPhaseSpan>& mulPhaseSpans() { static std::vector<MulPhaseSpan> v; return v; }

inline bool mulProfileEnabled() {
    static const bool on = getenv("ZISK_MUL_PROFILE") != nullptr;
    return on;
}

// Returns the span's index, or SIZE_MAX when profiling is off.
inline size_t mul_phase_begin(int phase, cudaStream_t stream, uint64_t air = 0, uint64_t jobs = 0,
                              uint64_t rows = 0) {
    if (!mulProfileEnabled()) return SIZE_MAX;
    MulPhaseSpan sp{nullptr, nullptr, phase, air, jobs, rows};
    CHECKCUDAERR(cudaEventCreate(&sp.start));
    CHECKCUDAERR(cudaEventCreate(&sp.stop));
    CHECKCUDAERR(cudaEventRecord(sp.start, stream));
    std::lock_guard<std::mutex> lk(mulEventMutex());
    mulPhaseSpans().push_back(sp);
    return mulPhaseSpans().size() - 1;
}

inline void mul_phase_end(size_t idx, cudaStream_t stream) {
    if (idx == SIZE_MAX) return;
    std::lock_guard<std::mutex> lk(mulEventMutex());
    CHECKCUDAERR(cudaEventRecord(mulPhaseSpans()[idx].stop, stream));
}

// Summed GPU time per phase, in ms, and the number of launches. Clears what it reports so a second
// proof in the same process measures only its own window.
inline void mul_phase_report() {
    if (!mulProfileEnabled()) return;
    std::vector<MulPhaseSpan> take;
    {
        std::lock_guard<std::mutex> lk(mulEventMutex());
        take.swap(mulPhaseSpans());
    }
    double total[MUL_PHASE_N] = {0.0, 0.0};
    uint64_t count[MUL_PHASE_N] = {0, 0};
    // Per (phase, air): time, launches, jobs, rows -- so the report says WHERE the time goes.
    std::map<std::pair<int, uint64_t>, std::tuple<double, uint64_t, uint64_t, uint64_t>> byAir;
    for (MulPhaseSpan& sp : take) {
        float ms = 0.0f;
        // A span whose stop never recorded (an aborted job) is skipped rather than charged.
        if (cudaEventQuery(sp.stop) == cudaSuccess && cudaEventElapsedTime(&ms, sp.start, sp.stop) == cudaSuccess) {
            total[sp.phase] += ms;
            count[sp.phase] += 1;
            auto& e = byAir[{sp.phase, sp.air}];
            std::get<0>(e) += ms; std::get<1>(e) += 1;
            std::get<2>(e) = sp.jobs; std::get<3>(e) = sp.rows;
        }
        cudaEventDestroy(sp.start);
        cudaEventDestroy(sp.stop);
    }
    std::vector<std::pair<double, std::string>> lines;
    for (auto& [k, v] : byAir) {
        char lb[200];
        snprintf(lb, sizeof(lb), "  %s air %llu:%llu  %.1f ms over %llu launches, %llu jobs (%llu mapped), %llu job-rows",
                 k.first == MUL_PHASE_KERNEL ? "kernel" : "interp",
                 (unsigned long long)(k.second >> 32), (unsigned long long)(k.second & 0xffffffffULL),
                 std::get<0>(v), (unsigned long long)std::get<1>(v),
                 (unsigned long long)(std::get<2>(v) & 0xffffffffULL),
                 (unsigned long long)(std::get<2>(v) >> 32),
                 (unsigned long long)std::get<3>(v));
        lines.emplace_back(std::get<0>(v), std::string(lb));
    }
    std::sort(lines.begin(), lines.end(), [](auto& a, auto& b) { return a.first > b.first; });
    char buf[256];
    snprintf(buf, sizeof(buf),
             "Multiplicity GPU: scatter kernel %.1f ms (%llu launches), interpreter %.1f ms (%llu launches), total %.1f ms",
             total[MUL_PHASE_KERNEL], (unsigned long long)count[MUL_PHASE_KERNEL],
             total[MUL_PHASE_INTERP], (unsigned long long)count[MUL_PHASE_INTERP],
             total[MUL_PHASE_KERNEL] + total[MUL_PHASE_INTERP]);
    zklog.info(buf);
    for (auto& l : lines) zklog.info(l.second);
}

inline void mul_wait_scatters(int gpuId) {
    std::vector<cudaEvent_t> take;
    {
        std::lock_guard<std::mutex> lk(mulEventMutex());
        auto it = mulEventPools().find(gpuId);
        if (it == mulEventPools().end()) return;
        take.swap(it->second.pending);
    }
    for (cudaEvent_t e : take) CHECKCUDAERR(cudaEventSynchronize(e));
    std::lock_guard<std::mutex> lk(mulEventMutex());
    MulEventPool& p = mulEventPools()[gpuId];
    p.freelist.insert(p.freelist.end(), take.begin(), take.end());
}

// Persistent per-device mirror. NOT a slice of d_aux_trace: that is per-stream scratch.
struct MulAcc {
    uint64_t* d_acc = nullptr;
    uint64_t  n_counters = 0;
    int       gpuId = -1;
};

inline MulAcc* mul_acc_init(int gpuId, uint64_t n_counters) {
    CHECKCUDAERR(cudaSetDevice(gpuId));
    MulAcc* a = new MulAcc();
    a->gpuId = gpuId;
    a->n_counters = n_counters;
    if (cudaMalloc(&a->d_acc, n_counters * sizeof(uint64_t)) != cudaSuccess) {
        delete a;
        return nullptr;
    }
    mulMemsetSync(gpuId, a->d_acc, 0, n_counters * sizeof(uint64_t));
    return a;
}

// One mirror per (air, gpu), only for airs hosting a migrated table.
inline std::map<std::pair<uint64_t,int>, MulAcc*>& mulAccs() {
    static std::map<std::pair<uint64_t,int>, MulAcc*> m;
    return m;
}

// True iff a registered decoder targets a table in this layout.
inline bool mulLayoutHostsMigrated(const MulVtLayout& L) {
    for (const auto& d : mulDecoders())
        if (L.accBase.count(d.table_id)) return true;
    return false;
}

// Allocate mirrors for every air that hosts a migrated table. Idempotent.
inline void mul_alloc_devices(const int* gpuIds, int nGpus) {
    for (const auto& L : mulVtLayouts()) {
        if (!mulLayoutHostsMigrated(L)) continue;
        for (int g = 0; g < nGpus; ++g) {
            auto key = std::make_pair(L.airId, gpuIds[g]);
            if (mulAccs().count(key)) continue;
            MulAcc* a = mul_acc_init(gpuIds[g], L.nCounters);
            if (a == nullptr) {
                zklog.error("multiplicity accumulator alloc failed: "
                            + std::to_string(L.nCounters * sizeof(uint64_t) / 1000000) + " MB");
                exitProcess();
            }
            mulAccs()[key] = a;
            // Per (air, gpu) -- superseded, for an operator, by the one aggregate per-device
            // residency line mul_alloc logs after every allocation for that device completes.
            zklog.trace("Multiplicity accumulator: "
                       + std::to_string(L.nCounters * sizeof(uint64_t) / 1000000)
                       + " MB for air " + std::to_string(L.airId)
                       + " on GPU " + std::to_string(gpuIds[g]));
        }
    }
}

// Each table's key->row index, mirrored per device. Allocated once alongside the accumulators and
// never freed: it is setup-derived and outlives every proof.
inline std::map<std::pair<uint64_t,int>, uint32_t*>& mulIndexDev() {
    static std::map<std::pair<uint64_t,int>, uint32_t*> m;
    return m;
}

inline std::map<std::pair<uint64_t,int>, uint64_t*>& mulMapDev() {
    static std::map<std::pair<uint64_t,int>, uint64_t*> m;
    return m;
}

inline void mul_alloc_maps(int gpuId) {
    CHECKCUDAERR(cudaSetDevice(gpuId));
    for (const auto& kv : mulTableMaps()) {
        auto key = std::make_pair(kv.first, gpuId);
        if (mulMapDev().count(key) || kv.second.empty()) continue;
        const size_t bytes = kv.second.size() * sizeof(uint64_t);
        uint64_t* d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess) {
            zklog.error("multiplicity: could not allocate the map for table "
                        + std::to_string(kv.first) + " (" + std::to_string(bytes / 1000000) + " MB)");
            exitProcess();
        }
        mulCopySync(gpuId, d, kv.second.data(), bytes, cudaMemcpyHostToDevice);
        mulMapDev()[key] = d;
        // Per (table, gpu) -- superseded, for an operator, by the aggregate per-device residency
        // line mul_alloc logs once every table for that device has been mirrored.
        zklog.trace("Multiplicity map: table " + std::to_string(kv.first) + " mirrored on gpu "
                   + std::to_string(gpuId) + " (" + std::to_string(bytes / 1000000) + " MB)");
    }
}

inline std::map<std::pair<uint64_t,int>, uint64_t*>& mulDigitDev() {
    static std::map<std::pair<uint64_t,int>, uint64_t*> m;
    return m;
}

inline void mul_alloc_digits(int gpuId) {
    CHECKCUDAERR(cudaSetDevice(gpuId));
    for (const auto& kv : mulTableDigits()) {
        auto key = std::make_pair(kv.first, gpuId);
        if (mulDigitDev().count(key) || kv.second.empty()) continue;
        const size_t bytes = kv.second.size() * sizeof(uint64_t);
        uint64_t* d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess) {
            zklog.error("multiplicity: could not allocate the digit table for table "
                        + std::to_string(kv.first));
            exitProcess();
        }
        mulCopySync(gpuId, d, kv.second.data(), bytes, cudaMemcpyHostToDevice);
        mulDigitDev()[key] = d;
    }
}

// Indexed-base tables (table 125's shape): the base array mirrored per device, plus a small
// device-resident MulDecoder shell whose baseTab already points at that mirror. A job's `dec`
// pointer becomes this shell's device address (see mulPlanDevice in multiplicity_kernel.cuh) --
// the kernel never dereferences the host struct in mulDecoders().
inline std::map<std::pair<uint64_t,int>, uint64_t*>& mulIndexedBaseTabDev() {
    static std::map<std::pair<uint64_t,int>, uint64_t*> m;
    return m;
}
inline std::map<std::pair<uint64_t,int>, MulDecoder*>& mulIndexedBaseDecDev() {
    static std::map<std::pair<uint64_t,int>, MulDecoder*> m;
    return m;
}

inline void mul_alloc_indexed_base(int gpuId) {
    CHECKCUDAERR(cudaSetDevice(gpuId));
    for (const auto& kv : mulTableIndexedBase()) {
        auto key = std::make_pair(kv.first, gpuId);
        if (mulIndexedBaseDecDev().count(key) || kv.second.empty()) continue;
        const size_t bytes = kv.second.size() * sizeof(uint64_t);
        uint64_t* dBase = nullptr;
        if (cudaMalloc(&dBase, bytes) != cudaSuccess) {
            zklog.error("multiplicity: could not allocate the indexed-base table for table "
                        + std::to_string(kv.first) + " (" + std::to_string(bytes / 1000000) + " MB)");
            exitProcess();
        }
        mulCopySync(gpuId, dBase, kv.second.data(), bytes, cudaMemcpyHostToDevice);
        mulIndexedBaseTabDev()[key] = dBase;

        // The shell carries only the selector/stride shape -- small and fixed-size -- with baseTab
        // repointed at this GPU's mirror of the (potentially large) base array.
        MulDecoder shell{};
        const MulDecoder* host = mulDecoderFor(kv.first);
        if (host != nullptr) {
            shell.nSel = host->nSel;
            for (uint32_t i = 0; i < host->nSel; ++i) {
                shell.selCol[i] = host->selCol[i];
                shell.selShift[i] = host->selShift[i];
                shell.selMask[i] = host->selMask[i];
            }
            shell.nStride = host->nStride;
            for (uint32_t i = 0; i < host->nStride; ++i) {
                shell.strideCol[i] = host->strideCol[i];
                shell.strideVal[i] = host->strideVal[i];
            }
        }
        shell.baseTab = dBase;
        MulDecoder* dDec = nullptr;
        if (cudaMalloc(&dDec, sizeof(MulDecoder)) != cudaSuccess) {
            zklog.error("multiplicity: could not allocate the indexed-base decoder shell for table "
                        + std::to_string(kv.first));
            exitProcess();
        }
        mulCopySync(gpuId, dDec, &shell, sizeof(MulDecoder), cudaMemcpyHostToDevice);
        mulIndexedBaseDecDev()[key] = dDec;
        // Per (table, gpu) -- same reasoning as the map registration above: trace, not info.
        zklog.trace("Multiplicity indexed-base: table " + std::to_string(kv.first) + " mirrored on gpu "
                   + std::to_string(gpuId) + " (" + std::to_string(bytes / 1000000) + " MB)");
    }
}

inline const MulDecoder* mulIndexedBaseFor(uint64_t tableId, int gpuId) {
    auto it = mulIndexedBaseDecDev().find(std::make_pair(tableId, gpuId));
    return it == mulIndexedBaseDecDev().end() ? nullptr : it->second;
}

// The base array's own device mirror, for callers that patch a MulDecoder's baseTab field directly
// (a copy-by-value decoder, e.g. the interpreter fallback's Dest::scatter.dec) rather than swap in
// the whole device-resident shell mulIndexedBaseFor returns.
inline const uint64_t* mulIndexedBaseTabFor(uint64_t tableId, int gpuId) {
    auto it = mulIndexedBaseTabDev().find(std::make_pair(tableId, gpuId));
    return it == mulIndexedBaseTabDev().end() ? nullptr : it->second;
}

inline const uint64_t* mulDigitsFor(uint64_t tableId, int gpuId) {
    auto it = mulDigitDev().find(std::make_pair(tableId, gpuId));
    return it == mulDigitDev().end() ? nullptr : it->second;
}

inline const uint64_t* mulMapFor(uint64_t tableId, int gpuId) {
    auto it = mulMapDev().find(std::make_pair(tableId, gpuId));
    return it == mulMapDev().end() ? nullptr : it->second;
}

// Total bytes this device currently holds for maps, digit tables, indexed-base tables and
// accumulators -- the one number that actually matters to an operator here, since it is
// per-GPU and competes directly with the prover's own arena on that device.
inline uint64_t mul_gpu_resident_bytes(int gpuId) {
    uint64_t bytes = 0;
    for (const auto& kv : mulMapDev())
        if (kv.first.second == gpuId) bytes += mulTableMaps()[kv.first.first].size() * sizeof(uint64_t);
    for (const auto& kv : mulDigitDev())
        if (kv.first.second == gpuId) bytes += mulTableDigits()[kv.first.first].size() * sizeof(uint64_t);
    for (const auto& kv : mulIndexedBaseTabDev())
        if (kv.first.second == gpuId) bytes += mulTableIndexedBase()[kv.first.first].size() * sizeof(uint64_t);
    for (const auto& kv : mulAccs())
        if (kv.first.second == gpuId) bytes += kv.second->n_counters * sizeof(uint64_t);
    return bytes;
}

// Out-of-range decode record, one per device: a device pointer is only valid on the device that
// allocated it. Populated by `mul_alloc_oob` while single-threaded, then read-only, which is what
// makes the lookup safe from the many threads that commit instances concurrently.
inline std::map<int, uint64_t*>& mulOobMap() { static std::map<int, uint64_t*> m; return m; }

inline uint64_t* mulOob(int gpuId) {
    auto it = mulOobMap().find(gpuId);
    return it == mulOobMap().end() ? nullptr : it->second;
}

inline void mul_alloc_oob(int gpuId) {
    CHECKCUDAERR(cudaSetDevice(gpuId));
    uint64_t*& oob = mulOobMap()[gpuId];
    if (oob == nullptr) {
        CHECKCUDAERR(cudaMalloc(&oob, MUL_OOB_SLOTS * sizeof(uint64_t)));
        mulMemsetSync(gpuId, oob, 0, MUL_OOB_SLOTS * sizeof(uint64_t));
    }
}

// Once per proof, from ProofMan::reset -- the same per-proof boundary as the stream reset.
inline void mul_reset_all() {
    mul_reset_commits();
    for (auto& kv : mulOobMap()) {
        if (kv.second == nullptr) continue;
        CHECKCUDAERR(cudaSetDevice(kv.first));
        mulMemsetSync(kv.first, kv.second, 0, MUL_OOB_SLOTS * sizeof(uint64_t));
    }
    for (auto& kv : mulAccs()) {
        CHECKCUDAERR(cudaSetDevice(kv.second->gpuId));
        mulMemsetSync(kv.second->gpuId, kv.second->d_acc, 0, kv.second->n_counters * sizeof(uint64_t));
    }
}

// One span into the Rust accumulator (Vec<AtomicU64>, layout-compatible). += , not = : each GPU
// holds only the counts from the instances it committed.
inline void mul_acc_fold_span(const MulAcc* a, uint64_t* host_acc,
                              uint64_t base, uint64_t n, uint64_t* staging) {
    CHECKCUDAERR(cudaSetDevice(a->gpuId));
    mulCopySync(a->gpuId, staging, a->d_acc + base, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    for (uint64_t i = 0; i < n; ++i) {
        if (staging[i]) {
            reinterpret_cast<std::atomic<uint64_t>*>(&host_acc[base + i])
                ->fetch_add(staging[i], std::memory_order_relaxed);
        }
    }
}

// Decodes that landed outside their table's span. A correct set of decoders reports zero; anything
// else means a wrong decoder, and the counts already folded are wrong too. Read once per proof
// rather than per hint, which would sync the commit pipeline.
inline uint64_t mul_oob_report() {
    uint64_t total = 0;
    for (const auto& kv : mulOobMap()) {
        if (kv.second == nullptr) continue;
        uint64_t v[MUL_OOB_SLOTS] = {0};
        CHECKCUDAERR(cudaSetDevice(kv.first));
        mulCopySync(kv.first, v, kv.second, sizeof(v), cudaMemcpyDeviceToHost);
        if (v[0] == 0) continue;
        total += v[0];
        // Slots 4..(4+n-1) hold the key that missed, n = v[15]; key[0] is duplicated at v[3] for
        // whoever only wants the old single-value summary.
        std::string keyStr;
        for (uint64_t c = 0; c < v[MUL_OOB_SLOTS - 1]; ++c)
            keyStr += (c ? "," : "") + std::to_string(v[4 + c]);
        zklog.error("multiplicity: " + std::to_string(v[0]) + " decodes outside table "
                    + std::to_string(v[1]) + " (first from air " + std::to_string(v[2] >> 32) + "/"
                    + std::to_string(v[2] & 0xFFFFFFFF) + ", key=[" + keyStr
                    + "]) -- that decoder is wrong");
    }
    return total;
}

// Fold every prover-owned span of `airId` into the Rust accumulator. One pass per GPU: each device
// counted only the instances that ran on it, so the spans add rather than replace.
void mul_transpose_acc_launch(const uint64_t* acc, uint64_t* trace, uint64_t numRows, uint64_t nCols,
                              cudaStream_t stream);
void mul_acc_add_launch(uint64_t* dst, const uint64_t* src, uint64_t n, cudaStream_t stream);

// Peer staging buffer, one per GPU, allocated only when a second GPU exists.
inline std::map<int, uint64_t*>& mulPeerStage() { static std::map<int, uint64_t*> m; return m; }

// Write air `airId`'s counts straight into the committed trace at `dst`, on the device.
//
// This is what replaces the host round-trip: without it the accumulator is copied to the host,
// added into a second host array, transposed there and uploaded again -- four passes over the same
// ~370 MB, none of which the counts need. Returns false when it cannot own the export (no
// accumulator, or a shape it does not recognise), and the caller must then take the host path,
// which is still what a cross-rank reduction needs.
// True when every table of `airId` is prover-owned, i.e. the device can produce the whole cm1.
inline bool& mulDeviceExportEnabled() { static bool on = false; return on; }

inline bool mul_air_fully_owned(uint64_t airId) {
    if (!mulDeviceExportEnabled()) return false;
    const MulVtLayout* L = nullptr;
    for (const auto& l : mulVtLayouts()) if (l.airId == airId) { L = &l; break; }
    if (L == nullptr || L->accBase.empty()) return false;
    for (const auto& kv : L->accBase) if (mulDecoderFor(kv.first) == nullptr) return false;
    return !mulAccs().empty();
}

inline bool mul_export_to_trace(uint64_t airId, int gpuId, uint64_t* dst,
                                uint64_t numRows, uint64_t nCols, cudaStream_t stream) {
    if (dst == nullptr) return false;
    const MulVtLayout* L = nullptr;
    for (const auto& l : mulVtLayouts()) if (l.airId == airId) { L = &l; break; }
    if (L == nullptr) return false;

    // The accumulator is sized numRows * num_muls by construction; if the trace disagrees the
    // layouts have diverged and a silent transpose would write the wrong counts.
    if (numRows * nCols != L->nCounters) {
        zklog.error("multiplicity: air " + std::to_string(airId) + " trace is " + std::to_string(numRows)
                    + "x" + std::to_string(nCols) + " but its accumulator holds "
                    + std::to_string(L->nCounters) + " counters");
        return false;
    }

    // All or nothing: the transpose writes the whole cm1, so a table this air holds that the std
    // still counts would be overwritten with zeros. Refuse the device path rather than lose it.
    for (const auto& kv : L->accBase) {
        if (mulDecoderFor(kv.first) == nullptr) return false;
    }

    MulAcc* local = nullptr;
    std::vector<MulAcc*> remote;
    for (auto& kv : mulAccs()) {
        if (kv.first.first != airId) continue;
        if (kv.first.second == gpuId) local = kv.second;
        else remote.push_back(kv.second);
    }
    if (local == nullptr) return false;

    CHECKCUDAERR(cudaSetDevice(gpuId));
    // Every scatter that fed this air, on every device: their counts must be visible before the
    // transpose reads them.
    mul_wait_scatters(gpuId);
    for (MulAcc* r : remote) { CHECKCUDAERR(cudaSetDevice(r->gpuId)); mul_wait_scatters(r->gpuId); }
    CHECKCUDAERR(cudaSetDevice(gpuId));

    // Several GPUs each hold a partial: fold them onto this one first. Staged through a local
    // buffer rather than read across the link, so this does not depend on peer access being
    // enabled between every pair.
    if (!remote.empty()) {
        uint64_t*& stage = mulPeerStage()[gpuId];
        if (stage == nullptr)
            CHECKCUDAERR(cudaMalloc(&stage, L->nCounters * sizeof(uint64_t)));
        for (MulAcc* r : remote) {
            CHECKCUDAERR(cudaMemcpyPeerAsync(stage, gpuId, r->d_acc, r->gpuId,
                                             L->nCounters * sizeof(uint64_t), stream));
            mul_acc_add_launch(local->d_acc, stage, L->nCounters, stream);
        }
    }

    mul_transpose_acc_launch(local->d_acc, dst, numRows, nCols, stream);
    return true;
}

inline void mul_fold_air(uint64_t airId, uint64_t* host_acc) {
    const MulVtLayout* L = nullptr;
    for (const auto& l : mulVtLayouts()) if (l.airId == airId) { L = &l; break; }
    if (L == nullptr) return;

    std::vector<uint64_t> staging;
    for (auto& kv : mulAccs()) {
        if (kv.first.first != airId) continue;
        // Every instance that scatters into this air has committed by now, but on its own
        // stream, so the counts are not necessarily visible yet. Wait on the events the scatters
        // recorded rather than on the device: a device-wide sync is illegal while another thread
        // is capturing a graph, which happens constantly alongside this fold.
        CHECKCUDAERR(cudaSetDevice(kv.second->gpuId));
        mul_wait_scatters(kv.second->gpuId);
        for (const auto& d : mulDecoders()) {
            if (!L->accBase.count(d.table_id)) continue;
            if (staging.size() < d.n_rows) staging.resize(d.n_rows);
            mul_acc_fold_span(kv.second, host_acc, d.acc_base, d.n_rows, staging.data());
        }
    }
}

#endif
