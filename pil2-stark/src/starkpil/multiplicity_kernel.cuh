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

void mul_scatter_launch_rows(const MulJobDev* d_jobs, uint32_t nJobs, const uint64_t* const* bases_,
                             uint64_t domainSize, uint64_t maxRows, uint64_t* acc, uint64_t* oob,
                             uint64_t air, const MulInsnDev* d_prog, cudaStream_t stream);

// Scatter over a TILE of rows whose cm1 has been materialised separately (the streaming commit
// never holds a whole cm1). `traceRows` is the tile's height and rows are addressed tile-locally in
// it, while const pols and the uniform pools stay in their own, full-height layout -- so the two
// cannot share one domain size the way the ordinary launch does.
void mul_scatter_launch_tile(const MulJobDev* d_jobs, uint32_t nJobs, uint64_t rows,
                             uint64_t rowBegin, const uint64_t* const* bases, uint64_t traceRows,
                             uint64_t fullRows, uint64_t* acc, uint64_t* oob, uint64_t air,
                             cudaStream_t stream);

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
        if (e != cudaSuccess)
            zklog.error("multiplicity: could not allocate the scatter plan for air "
                        + std::to_string(airgroupId) + "/" + std::to_string(airId) + " ("
                        + std::to_string(jb) + " bytes): " + cudaGetErrorString(e)
                        + " -- every lookup this air feeds would go uncounted");
        else {
            // The host jobs carry HOST index pointers; a device job must point at this GPU's
            // mirror, so patch the copy rather than the plan (which every GPU shares).
            std::vector<MulJobDev> hj = plan.jobs;
            for (auto& j : hj) {
                if (j.digitCols != 0) {
                    j.digitTab = mulDigitsFor(j.tableId, gpuId);
                    if (j.digitTab == nullptr) {
                        zklog.error("multiplicity: table " + std::to_string(j.tableId)
                                    + " has a digit rule but no mirror on gpu " + std::to_string(gpuId));
                        exitProcess();
                    }
                }
                if (j.mapSlots != 0) {
                    j.mapKV = mulMapFor(j.tableId, gpuId);
                    if (j.mapKV == nullptr) {
                        zklog.error("multiplicity: table " + std::to_string(j.tableId)
                                    + " has a map but no mirror on gpu " + std::to_string(gpuId));
                        exitProcess();
                    }
                }
                if (j.hasIndexedBase) {
                    j.dec = mulIndexedBaseFor(j.tableId, gpuId);
                    if (j.dec == nullptr) {
                        zklog.error("multiplicity: table " + std::to_string(j.tableId)
                                    + " has an indexed-base rule but no mirror on gpu " + std::to_string(gpuId));
                        exitProcess();
                    }
                }
            }
            // Never on the default stream: this runs on the commit path, where other threads are
            // capturing graphs and a legacy-stream copy would fail and poison the capture.
            mulCopySync(gpuId, dj, hj.data(), jb, cudaMemcpyHostToDevice);
            d.jobs = dj;
        }
    }
    // The air's instruction buffer, shared by every job that carries a program. Failing to place it
    // is not recoverable: those jobs would evaluate an empty program and count nothing.
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
