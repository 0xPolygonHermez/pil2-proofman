#ifndef MULTIPLICITY_KERNEL_CUH
#define MULTIPLICITY_KERNEL_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include "multiplicity_job.hpp"
#include "multiplicity_plan.hpp"
#include "multiplicity.cuh"   // mulCopySync: these uploads must stay off the legacy stream

// Scatter rows [0, rows) of a `domainSize`-row trace. `bases` is indexed by MulSrc; a null entry is
// fine as long as no term names it. `packed` and the rest are slot-only: cm1 is read from the
// packed rows in place, and `table` and its geometry serve an indexed air.
void mul_scatter_launch(const MulJobDev* d_jobs, uint32_t nJobs, uint64_t rows, uint64_t domainSize,
                        const uint64_t* const* bases, uint64_t* acc, uint64_t* oob, uint64_t air,
                        const MulInsnDev* d_prog, cudaStream_t stream,
                        const uint64_t* packed = nullptr, uint64_t wordsPerRow = 0,
                        const uint64_t* side = nullptr, const uint64_t* table = nullptr,
                        uint64_t wordsPerEntry = 0, uint64_t numEntries = 0,
                        uint64_t indexBits = 0, uint32_t packedColMajor = 0);

// One device copy of the plan per (air, gpu), built on first use.
struct MulPlanDev { const MulJobDev* jobs = nullptr; const MulInsnDev* prog = nullptr; };

inline MulPlanDev mulPlanDevice(const MulPlan& plan, uint64_t airgroupId, uint64_t airId, int gpuId) {
    static std::map<std::tuple<uint64_t,uint64_t,int>, MulPlanDev> bufs;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::make_tuple(airgroupId, airId, gpuId);
    auto it = bufs.find(key);
    if (it != bufs.end()) return it->second;
    MulPlanDev d;
    if (!plan.jobs.empty()) {
        MulJobDev* dj = nullptr;
        const size_t jb = plan.jobs.size() * sizeof(MulJobDev);
        const cudaError_t e = cudaMalloc(&dj, jb);
        if (e != cudaSuccess) {
            zklog.error("multiplicity: could not allocate the scatter plan for air "
                        + std::to_string(airgroupId) + "/" + std::to_string(airId) + " ("
                        + std::to_string(jb) + " bytes): " + cudaGetErrorString(e)
                        + " -- every lookup this air feeds would go uncounted");
            exitProcess();
        }
        // Patch host index pointers to this GPU's mirror in the copy; the plan is shared.
        std::vector<MulJobDev> hj = plan.jobs;
        for (auto& j : hj) {
            if (j.mapSlots != 0) {
                j.mapKV = mulMapFor(j.tableId, gpuId);
                if (j.mapKV == nullptr) {
                    zklog.error("multiplicity: table " + std::to_string(j.tableId)
                                + " has a map but no mirror on gpu " + std::to_string(gpuId));
                    exitProcess();
                }
            }
        }
        // Never on the default stream: it would poison concurrent graph captures.
        mulCopySync(gpuId, dj, hj.data(), jb, cudaMemcpyHostToDevice);
        d.jobs = dj;
    }
    // The air's instruction buffer. Failure is fatal: jobs would silently count nothing.
    if (!plan.prog.empty()) {
        MulInsnDev* dp = nullptr;
        const size_t pb = plan.prog.size() * sizeof(MulInsnDev);
        const cudaError_t e = cudaMalloc(&dp, pb);
        if (e != cudaSuccess) {
            zklog.error("multiplicity: could not allocate the scatter program for air "
                        + std::to_string(airgroupId) + "/" + std::to_string(airId) + " ("
                        + std::to_string(pb) + " bytes): " + cudaGetErrorString(e)
                        + " -- every lookup it evaluates would go uncounted");
            exitProcess();
        }
        mulCopySync(gpuId, dp, plan.prog.data(), pb, cudaMemcpyHostToDevice);
        d.prog = dp;
    }
    bufs[key] = d;
    return d;
}

#endif
