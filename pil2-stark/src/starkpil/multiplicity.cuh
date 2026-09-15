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
            zklog.info("Multiplicity accumulator: "
                       + std::to_string(L.nCounters * sizeof(uint64_t) / 1000000)
                       + " MB for air " + std::to_string(L.airId)
                       + " on GPU " + std::to_string(gpuIds[g]));
        }
    }
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
        zklog.error("multiplicity: " + std::to_string(v[0]) + " decodes outside table "
                    + std::to_string(v[1]) + " (first from air " + std::to_string(v[2] >> 32) + "/"
                    + std::to_string(v[2] & 0xFFFFFFFF) + ", row " + std::to_string(v[3])
                    + ") -- that decoder is wrong");
    }
    return total;
}

// Fold every prover-owned span of `airId` into the Rust accumulator. One pass per GPU: each device
// counted only the instances that ran on it, so the spans add rather than replace.
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
