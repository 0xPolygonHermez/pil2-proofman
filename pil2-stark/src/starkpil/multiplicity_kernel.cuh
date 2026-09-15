#ifndef MULTIPLICITY_KERNEL_CUH
#define MULTIPLICITY_KERNEL_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include "multiplicity_job.hpp"
#include "multiplicity_plan.hpp"
#include "multiplicity.cuh"   // mulCopySync: these uploads must stay off the legacy stream

void mul_scatter_launch_rows(const MulJobDev* d_jobs, uint32_t nJobs, const uint64_t* const* bases_,
                             uint64_t domainSize, uint64_t maxRows, uint64_t* acc, uint64_t* oob,
                             uint64_t air, cudaStream_t stream);

// Scatter over a TILE of rows whose cm1 has been materialised separately (the streaming commit
// never holds a whole cm1). `traceRows` is the tile's height and rows are addressed tile-locally in
// it, while const pols and the uniform pools stay in their own, full-height layout -- so the two
// cannot share one domain size the way the ordinary launch does.
void mul_scatter_launch_tile(const MulJobDev* d_jobs, uint32_t nJobs, uint64_t rows,
                             uint64_t rowBegin, const uint64_t* const* bases, uint64_t traceRows,
                             uint64_t fullRows, uint64_t* acc, uint64_t* oob, uint64_t air,
                             cudaStream_t stream);

// One device copy of the plan per (air, gpu), built on first use.
struct MulPlanDev { const MulJobDev* jobs = nullptr; };

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
            // Never on the default stream: this runs on the commit path, where other threads are
            // capturing graphs and a legacy-stream copy would fail and poison the capture.
            mulCopySync(gpuId, dj, plan.jobs.data(), jb, cudaMemcpyHostToDevice);
            d.jobs = dj;
        }
    }
    bufs[key] = d;
    return d;
}

#endif
