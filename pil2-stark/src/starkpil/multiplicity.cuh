#ifndef MULTIPLICITY_CUH
#define MULTIPLICITY_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <atomic>
#include <vector>
#include <map>
#include <utility>
#include <mutex>
#include <thread>
#include <algorithm>
#include <string>
#include "zklog.hpp"
#include "exit_process.hpp"
#include "cuda_utils.cuh"
#include "multiplicity.hpp"

// ---------------------------------------------------------------------------------------------
// Every host<->device operation below runs on a per-device NON-BLOCKING stream. The legacy
// default stream would fail and poison any graph another thread is capturing concurrently.
inline cudaStream_t mulXferStream(int gpuId) {
    static std::map<int, cudaStream_t> m;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lk(mtx);
    auto it = m.find(gpuId);
    if (it != m.end()) return it->second;
    cudaStream_t s = nullptr;
    // A stream belongs to the device current at creation.
    int prev = 0;
    CHECKCUDAERR(cudaGetDevice(&prev));
    CHECKCUDAERR(cudaSetDevice(gpuId));
    CHECKCUDAERR(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CHECKCUDAERR(cudaSetDevice(prev));
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

// Each scatter re-records its stream's event and the fold waits on those, since
// cudaDeviceSynchronize is illegal while another thread captures. A stream runs in order, so its
// last record covers every scatter before it: one event per stream, created at warmup.
inline std::map<int, std::map<cudaStream_t, cudaEvent_t>>& mulStreamEvents() {
    static std::map<int, std::map<cudaStream_t, cudaEvent_t>> m;
    return m;
}
inline std::mutex& mulEventMutex() { static std::mutex m; return m; }

inline cudaEvent_t mulEventForLocked(int gpuId, cudaStream_t stream) {
    cudaEvent_t& e = mulStreamEvents()[gpuId][stream];
    if (e == nullptr) {
        // An event belongs to the device current at creation, and must match its stream's.
        int prev = 0;
        CHECKCUDAERR(cudaGetDevice(&prev));
        CHECKCUDAERR(cudaSetDevice(gpuId));
        CHECKCUDAERR(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        CHECKCUDAERR(cudaSetDevice(prev));
    }
    return e;
}

inline void mul_warm_stream_event(int gpuId, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(mulEventMutex());
    mulEventForLocked(gpuId, stream);
}

inline void mul_note_scatter(int gpuId, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(mulEventMutex());
    CHECKCUDAERR(cudaEventRecord(mulEventForLocked(gpuId, stream), stream));
}

inline void mul_wait_scatters(int gpuId) {
    std::vector<cudaEvent_t> events;
    {
        std::lock_guard<std::mutex> lk(mulEventMutex());
        auto it = mulStreamEvents().find(gpuId);
        if (it == mulStreamEvents().end()) return;
        for (const auto& se : it->second) events.push_back(se.second);
    }
    for (cudaEvent_t e : events) CHECKCUDAERR(cudaEventSynchronize(e));
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

// This device's accumulator, or null when nothing is prover-owned.
inline MulAcc* mulAccOnGpu(int gpuId) {
    if (mulDecoders().empty()) return nullptr;
    for (const auto& kv : mulAccs())
        if (kv.first.second == gpuId) return kv.second;
    return nullptr;
}

// Allocate mirrors for every air that hosts a migrated table. Idempotent.
inline void mul_alloc_devices(const int* gpuIds, int nGpus) {
    // The GPU scatter writes every job into its device's one accumulator (mulAccOnGpu) and ignores
    // MulJobDev::hostAirKey, so a second host air would get the first one's counts.
    std::string hosts;
    uint32_t nHosts = 0;
    for (const auto& L : mulVtLayouts())
        if (mulLayoutHostsMigrated(L)) { ++nHosts; hosts += " " + mulAirName(L.airKey); }
    if (nHosts > 1) {
        zklog.error("multiplicity: virtual-table airs" + hosts + " all host prover-owned tables, but "
                    "the GPU scatter supports one host air; keep the other airs' tables std-owned");
        exitProcess();
    }
    for (const auto& L : mulVtLayouts()) {
        if (!mulLayoutHostsMigrated(L)) continue;
        for (int g = 0; g < nGpus; ++g) {
            auto key = std::make_pair(L.airKey, gpuIds[g]);
            if (mulAccs().count(key)) continue;
            MulAcc* a = mul_acc_init(gpuIds[g], L.nCounters);
            if (a == nullptr) {
                zklog.error("multiplicity accumulator alloc failed: "
                            + std::to_string(L.nCounters * sizeof(uint64_t) / 1000000) + " MB");
                exitProcess();
            }
            mulAccs()[key] = a;
            // Per (air, gpu); mul_alloc logs the per-device aggregate.
            zklog.trace("Multiplicity accumulator: "
                       + std::to_string(L.nCounters * sizeof(uint64_t) / 1000000)
                       + " MB for air " + mulAirName(L.airKey)
                       + " on GPU " + std::to_string(gpuIds[g]));
        }
    }
}

// Each table's exact-match map, mirrored per device. Setup-derived, never freed.
inline std::map<std::pair<uint64_t,int>, uint64_t*>& mulMapDev() {
    static std::map<std::pair<uint64_t,int>, uint64_t*> m;
    return m;
}

// Mirror every table's map onto one GPU, once per (table, gpu). Failure is fatal, since the
// lookups it decodes would silently count nothing.
inline void mul_alloc_maps(int gpuId) {
    CHECKCUDAERR(cudaSetDevice(gpuId));
    for (const auto& kv : mulTableMaps()) {
        auto key = std::make_pair(kv.first, gpuId);
        const std::vector<uint64_t>& host = kv.second.kv;
        if (mulMapDev().count(key) || host.empty()) continue;
        const size_t bytes = host.size() * sizeof(uint64_t);
        uint64_t* d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess) {
            zklog.error("multiplicity: could not allocate the map for table " + std::to_string(kv.first)
                        + " (" + std::to_string(bytes / 1000000) + " MB)");
            exitProcess();
        }
        mulCopySync(gpuId, d, host.data(), bytes, cudaMemcpyHostToDevice);
        mulMapDev()[key] = d;
    }
}

inline const uint64_t* mulMapFor(uint64_t tableId, int gpuId) {
    auto it = mulMapDev().find(std::make_pair(tableId, gpuId));
    return it == mulMapDev().end() ? nullptr : it->second;
}

// Folding the other GPUs' partials into the exporting one. Per device: two streams, each with a
// staging chunk, so one chunk's peer copy overlaps the previous chunk's add; `done` marks the end
// of a device's pulls. Allocated by mul_alloc when there are several GPUs, never on the commit path.
static constexpr uint64_t MUL_PEER_CHUNK = 4ull << 20;   // counters per staging (32 MB)
struct MulPeer {
    cudaStream_t stream[2] = {};
    uint64_t* stage[2] = {};
    cudaEvent_t done[2] = {};
};
inline std::map<int, MulPeer>& mulPeers() { static std::map<int, MulPeer> m; return m; }

// Air -> the GPU whose accumulator holds this proof's total (the others keep their partials).
inline std::map<uint64_t, int>& mulFoldedOn() { static std::map<uint64_t, int> m; return m; }

inline void mul_alloc_peers(const std::vector<int>& gpuIds) {
    for (int a : gpuIds) {
        if (mulPeers().count(a)) continue;
        CHECKCUDAERR(cudaSetDevice(a));
        // Direct DMA between the cards where the topology allows it; the driver bounces through
        // the host otherwise.
        for (int b : gpuIds) {
            int can = 0;
            if (a == b || cudaDeviceCanAccessPeer(&can, a, b) != cudaSuccess || !can) continue;
            const cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CHECKCUDAERR(e);
            (void)cudaGetLastError();
        }
        MulPeer& p = mulPeers()[a];
        for (int k = 0; k < 2; k++) {
            CHECKCUDAERR(cudaStreamCreateWithFlags(&p.stream[k], cudaStreamNonBlocking));
            CHECKCUDAERR(cudaMalloc(&p.stage[k], MUL_PEER_CHUNK * sizeof(uint64_t)));
            CHECKCUDAERR(cudaEventCreateWithFlags(&p.done[k], cudaEventDisableTiming));
        }
    }
}

void mul_acc_add_launch(uint64_t* dst, const uint64_t* src, uint64_t n, cudaStream_t stream);

// Enqueue dst += src (add) or dst = src on dst's peer streams, after src's previous pulls.
inline void mulPull(const MulAcc* dst, const MulAcc* src, uint64_t n, bool add) {
    if (src == nullptr || !mulPeers().count(dst->gpuId) || !mulPeers().count(src->gpuId)) {
        zklog.error("multiplicity: no peer fold context between gpu " + std::to_string(dst->gpuId) +
                    " and gpu " + std::to_string(src ? src->gpuId : -1));
        exitProcess();
    }
    MulPeer& p = mulPeers()[dst->gpuId];
    const MulPeer& q = mulPeers()[src->gpuId];
    CHECKCUDAERR(cudaSetDevice(dst->gpuId));
    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++) CHECKCUDAERR(cudaStreamWaitEvent(p.stream[k], q.done[j], 0));
    if (!add) {
        CHECKCUDAERR(cudaMemcpyPeerAsync(dst->d_acc, dst->gpuId, src->d_acc, src->gpuId,
                                         n * sizeof(uint64_t), p.stream[0]));
    } else {
        for (uint64_t off = 0, c = 0; off < n; off += MUL_PEER_CHUNK, c++) {
            const int k = (int)(c & 1);
            const uint64_t len = std::min<uint64_t>(MUL_PEER_CHUNK, n - off);
            CHECKCUDAERR(cudaMemcpyPeerAsync(p.stage[k], dst->gpuId, src->d_acc + off, src->gpuId,
                                             len * sizeof(uint64_t), p.stream[k]));
            mul_acc_add_launch(dst->d_acc + off, p.stage[k], len, p.stream[k]);
        }
    }
    for (int k = 0; k < 2; k++) CHECKCUDAERR(cudaEventRecord(p.done[k], p.stream[k]));
}

// Total bytes this device holds for maps, accumulators and the peer staging.
inline uint64_t mul_gpu_resident_bytes(int gpuId) {
    uint64_t bytes = 0;
    for (const auto& kv : mulMapDev())
        if (kv.first.second == gpuId) bytes += mulTableMaps()[kv.first.first].kv.size() * sizeof(uint64_t);
    for (const auto& kv : mulAccs())
        if (kv.first.second == gpuId) bytes += kv.second->n_counters * sizeof(uint64_t);
    if (mulPeers().count(gpuId)) bytes += 2 * MUL_PEER_CHUNK * sizeof(uint64_t);
    return bytes;
}

// Out-of-range decode record, one per device. Populated by `mul_alloc_oob` while single-threaded,
// then read-only, so concurrent committers can read it without locking.
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

// Once per proof, from ProofMan::reset.
inline void mul_reset_all() {
    mul_reset_commits();
    mulFoldedOn().clear();
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

// Decodes outside their table's span; nonzero means a wrong decoder and wrong counts. Read once
// per proof, not per hint, to avoid syncing the commit pipeline.
inline uint64_t mul_oob_report() {
    uint64_t total = 0;
    for (const auto& kv : mulOobMap()) {
        if (kv.second == nullptr) continue;
        uint64_t v[MUL_OOB_SLOTS] = {0};
        CHECKCUDAERR(cudaSetDevice(kv.first));
        // Commits count once their scatter is enqueued, so it may still be writing the record.
        mul_wait_scatters(kv.first);
        mulCopySync(kv.first, v, kv.second, sizeof(v), cudaMemcpyDeviceToHost);
        if (v[0] == 0) continue;
        total += v[0];
        // Slots 4..(4+n-1) hold the missed key, n = v[15]; key[0] is also at v[3].
        std::string keyStr;
        for (uint64_t c = 0; c < v[MUL_OOB_SLOTS - 1]; ++c)
            keyStr += (c ? "," : "") + std::to_string(v[4 + c]);
        const uint64_t air = v[2] & ~MUL_OOB_SELECTOR;
        const std::string first = (v[2] & MUL_OOB_SELECTOR) ? "a selector " + std::to_string(v[3]) + " >= 2^32"
                                                              : "key=[" + keyStr + "] outside it";
        zklog.error("multiplicity: " + std::to_string(v[0]) + " bad lookups into table " + std::to_string(v[1])
                    + " (first from air " + std::to_string(air >> 32) + "/" + std::to_string(air & 0xFFFFFFFF)
                    + ": " + first + ")");
    }
    return total;
}

void mul_transpose_acc_launch(const uint64_t* acc, uint64_t* trace, uint64_t numRows, uint64_t nCols,
                              cudaStream_t stream);

// Set by mul_set_device_export: off unless the counts need no cross-rank reduction.
inline bool& mulDeviceExportEnabled() { static bool on = false; return on; }

// The layout of `airKey` when every table of it is prover-owned, i.e. the device can produce the
// whole cm1; else null.
inline const MulVtLayout* mulFullyOwnedLayout(uint64_t airKey) {
    if (!mulDeviceExportEnabled() || mulAccs().empty()) return nullptr;
    const MulVtLayout* L = mulLayoutFor(airKey);
    if (L == nullptr || L->accBase.empty()) return nullptr;
    for (const auto& kv : L->accBase) if (mulDecoderFor(kv.first) == nullptr) return nullptr;
    return L;
}

inline bool mul_air_fully_owned(uint64_t airKey) { return mulFullyOwnedLayout(airKey) != nullptr; }

// Write air `airKey`'s counts straight into the committed trace at `dst`, on the device.
// False when the air is not device-owned: the caller takes the host path, which a cross-rank
// reduction also needs. A device-owned air that cannot be exported is fatal: nothing fills its trace.
inline bool mul_export_to_trace(uint64_t airKey, int gpuId, uint64_t* dst,
                                uint64_t numRows, uint64_t nCols, cudaStream_t stream) {
    // Off with several ranks (the host reduces them) or when the std still counts a table here.
    // All or nothing: the transpose writes the whole cm1.
    const MulVtLayout* L = dst != nullptr ? mulFullyOwnedLayout(airKey) : nullptr;
    if (L == nullptr) return false;

    // The accumulator is numRows * num_muls; a mismatch means the layouts diverged.
    if (numRows * nCols != L->nCounters) {
        zklog.error("multiplicity: air " + mulAirName(airKey) + " trace is " + std::to_string(numRows)
                    + "x" + std::to_string(nCols) + " but its accumulator holds "
                    + std::to_string(L->nCounters) + " counters");
        exitProcess();
    }

    MulAcc* local = nullptr;
    std::vector<MulAcc*> remote;
    for (auto& kv : mulAccs()) {
        if (kv.first.first != airKey) continue;
        if (kv.first.second == gpuId) local = kv.second;
        else remote.push_back(kv.second);
    }
    if (local == nullptr) {
        zklog.error("multiplicity: air " + mulAirName(airKey) + " has no accumulator on gpu "
                    + std::to_string(gpuId));
        exitProcess();
    }

    CHECKCUDAERR(cudaSetDevice(gpuId));
    // Every scatter that fed this air, on every device, must be visible before the transpose.
    mul_wait_scatters(gpuId);
    for (MulAcc* r : remote) { CHECKCUDAERR(cudaSetDevice(r->gpuId)); mul_wait_scatters(r->gpuId); }
    CHECKCUDAERR(cudaSetDevice(gpuId));

    // First export of the proof: a tree reduction onto this GPU, pairs in parallel, log2(n) rounds.
    // Later ones (the proof, maybe on another GPU) copy the total from where it landed.
    if (!remote.empty()) {
        auto folded = mulFoldedOn().find(airKey);
        std::vector<const MulAcc*> used;
        if (folded == mulFoldedOn().end()) {
            std::vector<const MulAcc*> order{local};
            order.insert(order.end(), remote.begin(), remote.end());
            for (size_t step = 1; step < order.size(); step *= 2)
                for (size_t i = 0; i + step < order.size(); i += 2 * step)
                    mulPull(order[i], order[i + step], L->nCounters, true);
            used = order;
            mulFoldedOn()[airKey] = gpuId;
        } else if (folded->second != gpuId) {
            const MulAcc* holder = nullptr;
            for (const MulAcc* r : remote) if (r->gpuId == folded->second) holder = r;
            mulPull(local, holder, L->nCounters, false);
            used = {local, holder};
        }
        for (const MulAcc* a : used)
            for (int k = 0; k < 2; k++) CHECKCUDAERR(cudaStreamSynchronize(mulPeers()[a->gpuId].stream[k]));
        CHECKCUDAERR(cudaSetDevice(gpuId));
    }

    mul_transpose_acc_launch(local->d_acc, dst, numRows, nCols, stream);
    return true;
}

// Fold every prover-owned span of `airKey` into the Rust accumulator. The spans are cut into
// chunks that host threads take in turn; a thread copies its chunk from every GPU at once and adds
// them, so the GPUs' D2H copies overlap and the adds spread over cores. Each counter is still the
// sum of the same per-GPU values. Pinned staging and per-(thread, GPU) streams come from mul_alloc.
static constexpr uint64_t MUL_FOLD_CHUNK = 1ull << 19;   // counters per copy (4 MB)
static constexpr int MUL_FOLD_THREADS = 16;
struct MulFoldStaging {
    std::vector<int> gpuIds;
    std::vector<uint64_t*> ptr;                  // per thread: gpuIds.size() chunks
    std::vector<std::vector<cudaStream_t>> stream; // [thread][gpu index]
};
inline MulFoldStaging& mulFoldStaging() { static MulFoldStaging s; return s; }

inline void mul_alloc_fold_staging() {
    MulFoldStaging& s = mulFoldStaging();
    std::vector<int> ids;
    for (const auto& kv : mulAccs())
        if (std::find(ids.begin(), ids.end(), kv.first.second) == ids.end()) ids.push_back(kv.first.second);
    std::sort(ids.begin(), ids.end());
    if (ids.empty() || ids == s.gpuIds) return;
    int prev = 0;
    CHECKCUDAERR(cudaGetDevice(&prev));
    for (uint64_t* p : s.ptr) CHECKCUDAERR(cudaFreeHost(p));
    for (size_t t = 0; t < s.stream.size(); t++)
        for (size_t g = 0; g < s.stream[t].size(); g++) {
            CHECKCUDAERR(cudaSetDevice(s.gpuIds[g]));
            CHECKCUDAERR(cudaStreamDestroy(s.stream[t][g]));
        }
    s.gpuIds = ids;
    s.ptr.assign(MUL_FOLD_THREADS, nullptr);
    s.stream.assign(MUL_FOLD_THREADS, std::vector<cudaStream_t>(ids.size(), nullptr));
    // cudaMallocHost needs a current device: use one of this process's GPUs.
    CHECKCUDAERR(cudaSetDevice(ids[0]));
    for (int t = 0; t < MUL_FOLD_THREADS; t++) {
        CHECKCUDAERR(cudaMallocHost((void**)&s.ptr[t], ids.size() * MUL_FOLD_CHUNK * sizeof(uint64_t)));
        for (size_t g = 0; g < ids.size(); g++) {
            CHECKCUDAERR(cudaSetDevice(ids[g]));
            CHECKCUDAERR(cudaStreamCreateWithFlags(&s.stream[t][g], cudaStreamNonBlocking));
        }
    }
    // Only back to one of ours (see mul_fold_air).
    CHECKCUDAERR(cudaSetDevice(std::find(ids.begin(), ids.end(), prev) != ids.end() ? prev : ids[0]));
}

inline void mul_fold_air(uint64_t airKey, uint64_t* host_acc) {
    const MulFoldStaging& staging = mulFoldStaging();
    std::vector<std::pair<const MulAcc*, size_t>> accs;   // (accumulator, index into staging.gpuIds)
    for (auto& kv : mulAccs()) {
        if (kv.first.first != airKey) continue;
        auto it = std::find(staging.gpuIds.begin(), staging.gpuIds.end(), kv.second->gpuId);
        if (it == staging.gpuIds.end()) {
            zklog.error("multiplicity: no fold staging for gpu " + std::to_string(kv.second->gpuId) +
                        " (mul_alloc not run?)");
            exitProcess();
        }
        accs.emplace_back(kv.second, (size_t)(it - staging.gpuIds.begin()));
    }
    if (accs.empty()) return;
    // Restore the caller's device only if it is ours (see stage_witness_gpu).
    int prev = 0;
    CHECKCUDAERR(cudaGetDevice(&prev));
    const bool prevOwned = std::find(staging.gpuIds.begin(), staging.gpuIds.end(), prev) != staging.gpuIds.end();
    // Scatters committed on their own streams; wait on their events (no device-wide sync).
    for (const auto& a : accs) {
        CHECKCUDAERR(cudaSetDevice(a.first->gpuId));
        mul_wait_scatters(a.first->gpuId);
    }

    std::vector<std::pair<uint64_t, uint64_t>> spans;
    for (const auto& d : mulDecoders())
        if (d.hostAirKey == airKey && d.n_rows != 0) spans.emplace_back(d.acc_base, d.n_rows);
    std::sort(spans.begin(), spans.end());
    // Threads own disjoint chunks, so the spans must not overlap.
    for (size_t i = 1; i < spans.size(); i++)
        if (spans[i - 1].first + spans[i - 1].second > spans[i].first) {
            zklog.error("multiplicity: air " + mulAirName(airKey) + " has overlapping table spans");
            exitProcess();
        }
    std::vector<std::pair<uint64_t, uint64_t>> chunks;
    for (const auto& sp : spans)
        for (uint64_t off = 0; off < sp.second; off += MUL_FOLD_CHUNK)
            chunks.emplace_back(sp.first + off, std::min<uint64_t>(MUL_FOLD_CHUNK, sp.second - off));

    std::atomic<size_t> next{0};
    auto work = [&](int t) {
        uint64_t* stage = staging.ptr[t];
        for (size_t c; (c = next.fetch_add(1, std::memory_order_relaxed)) < chunks.size();) {
            const uint64_t base = chunks[c].first, n = chunks[c].second;
            for (size_t k = 0; k < accs.size(); k++) {
                const size_t g = accs[k].second;
                CHECKCUDAERR(cudaSetDevice(accs[k].first->gpuId));
                CHECKCUDAERR(cudaMemcpyAsync(stage + k * MUL_FOLD_CHUNK, accs[k].first->d_acc + base,
                                             n * sizeof(uint64_t), cudaMemcpyDeviceToHost, staging.stream[t][g]));
            }
            uint64_t* dst = host_acc + base;
            for (size_t k = 0; k < accs.size(); k++) {
                CHECKCUDAERR(cudaStreamSynchronize(staging.stream[t][accs[k].second]));
                const uint64_t* src = stage + k * MUL_FOLD_CHUNK;
                for (uint64_t i = 0; i < n; ++i) dst[i] += src[i];
            }
        }
    };
    const int nThreads = (int)std::min<size_t>(MUL_FOLD_THREADS, chunks.size());
    std::vector<std::thread> pool;
    for (int t = 1; t < nThreads; t++) pool.emplace_back(work, t);
    if (nThreads > 0) work(0);
    for (auto& th : pool) th.join();
    if (prevOwned) CHECKCUDAERR(cudaSetDevice(prev));
}

#endif
