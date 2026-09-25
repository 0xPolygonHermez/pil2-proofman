#ifndef MULTIPLICITY_CUH
#define MULTIPLICITY_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <atomic>
#include <vector>
#include <map>
#include <utility>
#include <mutex>
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

// Each scatter records an event on its stream and the fold waits on those, since
// cudaDeviceSynchronize is illegal while another thread captures.
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
    // MulJobDev::hostAirId, so a second host air would get the first one's counts.
    std::string hosts;
    uint32_t nHosts = 0;
    for (const auto& L : mulVtLayouts())
        if (mulLayoutHostsMigrated(L)) { ++nHosts; hosts += " " + std::to_string(L.airId); }
    if (nHosts > 1) {
        zklog.error("multiplicity: virtual-table airs" + hosts + " all host prover-owned tables, but "
                    "the GPU scatter supports one host air; keep the other airs' tables std-owned");
        exitProcess();
    }
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
            // Per (air, gpu); mul_alloc logs the per-device aggregate.
            zklog.trace("Multiplicity accumulator: "
                       + std::to_string(L.nCounters * sizeof(uint64_t) / 1000000)
                       + " MB for air " + std::to_string(L.airId)
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

// One span into the Rust accumulator (Vec<AtomicU64>, layout-compatible). Adds, since each GPU
// holds only its own instances' counts.
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

// Decodes outside their table's span; nonzero means a wrong decoder and wrong counts. Read once
// per proof, not per hint, to avoid syncing the commit pipeline.
inline uint64_t mul_oob_report() {
    uint64_t total = 0;
    for (const auto& kv : mulOobMap()) {
        if (kv.second == nullptr) continue;
        uint64_t v[MUL_OOB_SLOTS] = {0};
        CHECKCUDAERR(cudaSetDevice(kv.first));
        mulCopySync(kv.first, v, kv.second, sizeof(v), cudaMemcpyDeviceToHost);
        if (v[0] == 0) continue;
        total += v[0];
        // Slots 4..(4+n-1) hold the missed key, n = v[15]; key[0] is also at v[3].
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

void mul_transpose_acc_launch(const uint64_t* acc, uint64_t* trace, uint64_t numRows, uint64_t nCols,
                              cudaStream_t stream);

// Set by mul_set_device_export: off unless the counts need no cross-rank reduction.
inline bool& mulDeviceExportEnabled() { static bool on = false; return on; }

// The layout of `airId` when every table of it is prover-owned, i.e. the device can produce the
// whole cm1; else null.
inline const MulVtLayout* mulFullyOwnedLayout(uint64_t airId) {
    if (!mulDeviceExportEnabled() || mulAccs().empty()) return nullptr;
    const MulVtLayout* L = mulLayoutFor(airId);
    if (L == nullptr || L->accBase.empty()) return nullptr;
    for (const auto& kv : L->accBase) if (mulDecoderFor(kv.first) == nullptr) return nullptr;
    return L;
}

inline bool mul_air_fully_owned(uint64_t airId) { return mulFullyOwnedLayout(airId) != nullptr; }

// Write air `airId`'s counts straight into the committed trace at `dst`, on the device.
// Returns false when it cannot (no accumulator, unrecognised shape); the caller then takes the
// host path, which a cross-rank reduction also needs.
inline bool mul_export_to_trace(uint64_t airId, int gpuId, uint64_t* dst,
                                uint64_t numRows, uint64_t nCols, cudaStream_t stream) {
    // Off with several ranks (the host reduces them) or when the std still counts a table here.
    // All or nothing: the transpose writes the whole cm1.
    const MulVtLayout* L = dst != nullptr ? mulFullyOwnedLayout(airId) : nullptr;
    if (L == nullptr) return false;

    // The accumulator is numRows * num_muls; a mismatch means the layouts diverged.
    if (numRows * nCols != L->nCounters) {
        zklog.error("multiplicity: air " + std::to_string(airId) + " trace is " + std::to_string(numRows)
                    + "x" + std::to_string(nCols) + " but its accumulator holds "
                    + std::to_string(L->nCounters) + " counters");
        return false;
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
    // Every scatter that fed this air, on every device, must be visible before the transpose.
    mul_wait_scatters(gpuId);
    for (MulAcc* r : remote) { CHECKCUDAERR(cudaSetDevice(r->gpuId)); mul_wait_scatters(r->gpuId); }
    CHECKCUDAERR(cudaSetDevice(gpuId));

    // First export of the proof: a tree reduction onto this GPU, pairs in parallel, log2(n) rounds.
    // Later ones (the proof, maybe on another GPU) copy the total from where it landed.
    if (!remote.empty()) {
        auto folded = mulFoldedOn().find(airId);
        std::vector<const MulAcc*> used;
        if (folded == mulFoldedOn().end()) {
            std::vector<const MulAcc*> order{local};
            order.insert(order.end(), remote.begin(), remote.end());
            for (size_t step = 1; step < order.size(); step *= 2)
                for (size_t i = 0; i + step < order.size(); i += 2 * step)
                    mulPull(order[i], order[i + step], L->nCounters, true);
            used = order;
            mulFoldedOn()[airId] = gpuId;
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

// Fold every prover-owned span of `airId` into the Rust accumulator, adding one pass per GPU.
inline void mul_fold_air(uint64_t airId, uint64_t* host_acc) {
    std::vector<uint64_t> staging;
    for (auto& kv : mulAccs()) {
        if (kv.first.first != airId) continue;
        // Scatters committed on their own streams; wait on their events (no device-wide sync).
        CHECKCUDAERR(cudaSetDevice(kv.second->gpuId));
        mul_wait_scatters(kv.second->gpuId);
        for (const auto& d : mulDecoders()) {
            if (d.hostAirId != airId) continue;
            if (staging.size() < d.n_rows) staging.resize(d.n_rows);
            mul_acc_fold_span(kv.second, host_acc, d.acc_base, d.n_rows, staging.data());
        }
    }
}

#endif
