#ifndef __GL64_GPU_CUH__
#define __GL64_GPU_CUH__

#include <cstdint>
#include <cassert>
#include "goldilocks_base_field.hpp"
#include "gpu_timer.cuh"
#include <mutex>
#include "cuda_utils.cuh"
#include "transcriptGL.cuh"
#include "expressions_gpu.cuh"
#include <limits.h>
#include "gl64_t.cuh"

class gl64_gpu
{
public:
    // GPU utilities
    static __device__ __forceinline__ void copy_gpu(gl64_t *dst, const gl64_t *src, bool const_src)
    {
        int tid = const_src ? 0 : threadIdx.x;
        dst[threadIdx.x] = src[tid];
    }

    static __device__ __forceinline__ void copy_gpu(gl64_t *dst, uint64_t stride_dst, const gl64_t *src, bool const_src)
    {
        int tid = const_src ? 0 : threadIdx.x;
        dst[threadIdx.x * stride_dst] = src[tid];
    }

    static __device__ __forceinline__ void op_gpu(uint64_t op, gl64_t *c, const gl64_t *a, bool const_a, const gl64_t *b, bool const_b)
    {
        int tida = const_a ? 0 : threadIdx.x;
        int tidb = const_b ? 0 : threadIdx.x;

        switch (op)
        {
        case 0: c[threadIdx.x] = a[tida] + b[tidb]; break;
        case 1: c[threadIdx.x] = a[tida] - b[tidb]; break;
        case 2: c[threadIdx.x] = a[tida] * b[tidb]; break;
        case 3: c[threadIdx.x] = b[tidb] - a[tida]; break;
        default: assert(0); break;
        }
    }
};

struct AirInstanceInfo {
    uint64_t airgroupId;
    uint64_t airId;

    uint64_t const_pols_offset;
    uint64_t const_tree_offset;

    bool stored = false;

    ExpressionsGPU *expressions_gpu;
    int64_t *opening_points;

    uint64_t numBatchesEvals;
    EvalInfo **evalsInfo;
    uint64_t *evalsInfoSizes;

    EvalInfo **evalsInfoFRI;
    uint64_t *evalsInfoFRISizes;
    
    SetupCtx *setupCtx;

    Goldilocks::Element *verkeyRoot;

    uint64_t nStreams = 1;

    AirInstanceInfo(uint64_t airgroupId, uint64_t airId, SetupCtx *setupCtx, Goldilocks::Element *verkeyRoot_, uint64_t nStreams_): setupCtx(setupCtx), airgroupId(airgroupId), airId(airId), nStreams(nStreams_) {
        int64_t *d_openingPoints;
        CHECKCUDAERR(cudaMalloc(&d_openingPoints, setupCtx->starkInfo.openingPoints.size() * sizeof(int64_t)));
        CHECKCUDAERR(cudaMemcpy(d_openingPoints, setupCtx->starkInfo.openingPoints.data(), setupCtx->starkInfo.openingPoints.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        opening_points = d_openingPoints;
        expressions_gpu = new ExpressionsGPU(*setupCtx, setupCtx->starkInfo.nrowsPack, setupCtx->starkInfo.maxNBlocks);

        Goldilocks::Element *d_verkeyRoot;
        CHECKCUDAERR(cudaMalloc(&d_verkeyRoot, HASH_SIZE * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMemcpy(d_verkeyRoot, verkeyRoot_, HASH_SIZE * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
        verkeyRoot = d_verkeyRoot;

        uint64_t size_eval = setupCtx->starkInfo.evMap.size();
        uint64_t num_batches = (setupCtx->starkInfo.openingPoints.size() + 3) / 4;

        evalsInfo = new EvalInfo*[num_batches];
        evalsInfoSizes = new uint64_t[num_batches];
        numBatchesEvals = num_batches;

        uint64_t count = 0;
        for(uint64_t i = 0; i < setupCtx->starkInfo.openingPoints.size(); i += 4) {
            std::vector<int64_t> openingPoints;
            for(uint64_t j = 0; j < 4; ++j) {
                if(i + j < setupCtx->starkInfo.openingPoints.size()) {
                    openingPoints.push_back(setupCtx->starkInfo.openingPoints[i + j]);
                }
            }
            
            EvalInfo* evalsInfoHost = new EvalInfo[size_eval];

            uint64_t nEvals = 0;

            for (uint64_t k = 0; k < size_eval; k++)
            {
                EvMap ev = setupCtx->starkInfo.evMap[k];
                auto it = std::find(openingPoints.begin(), openingPoints.end(), ev.prime);
                bool containsOpening = it != openingPoints.end();
                if(!containsOpening) continue;
                string type = ev.type == EvMap::eType::cm ? "cm" : ev.type == EvMap::eType::custom ? "custom"
                                                                                                : "fixed";
                PolMap polInfo = type == "cm" ? setupCtx->starkInfo.cmPolsMap[ev.id] : type == "custom" ? setupCtx->starkInfo.customCommitsMap[ev.commitId][ev.id]
                                                                                                            : setupCtx->starkInfo.constPolsMap[ev.id];
                evalsInfoHost[nEvals].type = type == "cm" ? 0 : type == "custom" ? 1
                                                                        : 2; //rick: harcoded
                evalsInfoHost[nEvals].offset = setupCtx->starkInfo.getTraceOffset(type, polInfo, true);
                evalsInfoHost[nEvals].stride = setupCtx->starkInfo.getTraceNColsSection(type, polInfo, true);
                evalsInfoHost[nEvals].dim = polInfo.dim;
                evalsInfoHost[nEvals].openingPos = std::distance(openingPoints.begin(), it);
                evalsInfoHost[nEvals].evalPos = k;
                nEvals++;
            }

            EvalInfo* d_evalsInfo = nullptr;
            CHECKCUDAERR(cudaMalloc(&d_evalsInfo, nEvals * sizeof(EvalInfo)));
            CHECKCUDAERR(cudaMemcpy(d_evalsInfo, evalsInfoHost, nEvals * sizeof(EvalInfo), cudaMemcpyHostToDevice));

            evalsInfo[count] = d_evalsInfo;
            evalsInfoSizes[count] = nEvals;
            delete[] evalsInfoHost;
            count++;
        }

        uint64_t nOpeningPoints = setupCtx->starkInfo.openingPoints.size();

        evalsInfoFRI = new EvalInfo*[nOpeningPoints];
        evalsInfoFRISizes = new uint64_t[nOpeningPoints];

        std::fill(evalsInfoFRISizes, evalsInfoFRISizes + nOpeningPoints, 0);
        for (uint64_t i = 0; i < setupCtx->starkInfo.evMap.size(); i++) {
            evalsInfoFRISizes[setupCtx->starkInfo.evMap[i].openingPos]++;
        }

        EvalInfo** evalsInfoByOpeningPos = new EvalInfo*[nOpeningPoints];
        for (uint64_t pos = 0; pos < nOpeningPoints; pos++) {
            evalsInfoByOpeningPos[pos] = new EvalInfo[evalsInfoFRISizes[pos]];
        }

        std::fill(evalsInfoFRISizes, evalsInfoFRISizes + nOpeningPoints, 0);
        for (uint64_t i = 0; i < setupCtx->starkInfo.evMap.size(); i++) {
            EvMap ev = setupCtx->starkInfo.evMap[i];
            uint64_t pos = ev.openingPos;

            std::string type = (ev.type == EvMap::eType::cm) ? "cm" :
                            (ev.type == EvMap::eType::custom) ? "custom" : "fixed";

            PolMap polInfo = (type == "cm")      ? setupCtx->starkInfo.cmPolsMap[ev.id] :
                            (type == "custom")  ? setupCtx->starkInfo.customCommitsMap[ev.commitId][ev.id] :
                                                setupCtx->starkInfo.constPolsMap[ev.id];

            EvalInfo* evInfo = &evalsInfoByOpeningPos[pos][evalsInfoFRISizes[pos]];
            evInfo->type = (type == "cm") ? 0 : (type == "custom") ? 1 : 2;
            evInfo->offset = setupCtx->starkInfo.getTraceOffset(type, polInfo, true);
            evInfo->stride = setupCtx->starkInfo.getTraceNColsSection(type, polInfo, true);
            evInfo->dim = polInfo.dim;
            evInfo->evalPos = i;
            evInfo->openingPos = pos;

            evalsInfoFRISizes[pos]++;
        }

        for (uint64_t opening = 0; opening < nOpeningPoints; opening++) {
            CHECKCUDAERR(cudaMalloc(&evalsInfoFRI[opening], evalsInfoFRISizes[opening] * sizeof(EvalInfo)));
            CHECKCUDAERR(cudaMemcpy(evalsInfoFRI[opening], evalsInfoByOpeningPos[opening],
                                    evalsInfoFRISizes[opening] * sizeof(EvalInfo),
                                    cudaMemcpyHostToDevice));
            delete[] evalsInfoByOpeningPos[opening];
        }

        delete[] evalsInfoByOpeningPos;
    }

    ~AirInstanceInfo() {
        if (opening_points != nullptr) {
            CHECKCUDAERR(cudaFree(opening_points));
        }

        if (verkeyRoot != nullptr) {
            CHECKCUDAERR(cudaFree(verkeyRoot));
        }

        delete expressions_gpu;

        for (uint64_t i = 0; i < numBatchesEvals; ++i) {
            if (evalsInfo[i] != nullptr) {
                CHECKCUDAERR(cudaFree(evalsInfo[i]));
            }
        }

        delete[] evalsInfoSizes;
        delete[] evalsInfo;

        for (uint64_t i = 0; i < setupCtx->starkInfo.openingPoints.size(); ++i) {
            if (evalsInfoFRI[i] != nullptr) {
                CHECKCUDAERR(cudaFree(evalsInfoFRI[i]));
            }
        }

        delete[] evalsInfoFRISizes;
    }
};


struct StreamData{

    //const data
    cudaStream_t stream;
    uint32_t gpuId;
    uint32_t slotId;
    StepsParams *pinned_params;
    Goldilocks::Element *pinned_buffer;
    Goldilocks::Element *pinned_buffer_proof;
    Goldilocks::Element *pinned_buffer_exps_params;
    Goldilocks::Element *pinned_buffer_exps_args;

    uint64_t pinned_size = 256 * 1024 * 1024; //256MB, this is the size of pinned memory for consts, it can be changed if needed

    //runtime data
    uint32_t status; //0: unused, 1: loading, 2: full
    cudaEvent_t end_event;
    TimerGPU timer;

    TranscriptGL_GPU *transcript;
    TranscriptGL_GPU *transcript_helper;

    StepsParams *params;
    ExpsArguments *d_expsArgs;
    DestParamsGPU *d_destParams;

    //callback inputs
    void *root;
    void *pSetupCtx;
    uint64_t *proofBuffer; 
    string proofFile;
    uint64_t airgroupId; 
    uint64_t airId; 
    uint64_t instanceId;
    string proofType;
    
    uint64_t offset;
    
    bool recursive;
    bool extraStream;
    uint64_t streamsUsed;
    
    void initialize(uint64_t max_size_proof, uint32_t gpuId_, uint64_t offset_, bool recursive_){
        uint64_t maxExps = 1000; // TODO: CALCULATE IT PROPERLY!
        cudaSetDevice(gpuId_);
        CHECKCUDAERR(cudaStreamCreate(&stream));
        timer.init(stream);
        gpuId = gpuId_;
        offset = offset_;
        recursive = recursive_;
        cudaEventCreate(&end_event);
        status = 0;
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_buffer, pinned_size));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_buffer_proof, max_size_proof * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_buffer_exps_params, maxExps * 2 * sizeof(DestParamsGPU)));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_buffer_exps_args, maxExps * sizeof(ExpsArguments)));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_params, sizeof(StepsParams)));

        extraStream = false;
        streamsUsed = 1;
        root = nullptr;
        pSetupCtx = nullptr;
        proofBuffer = nullptr;
        airgroupId = UINT64_MAX;
        airId = UINT64_MAX;

        transcript = new TranscriptGL_GPU(3,
                                    true,
                                    stream);

        transcript_helper = new TranscriptGL_GPU(3,
                                           true,
                                           stream);

        CHECKCUDAERR(cudaMalloc(&params, sizeof(StepsParams)));
        CHECKCUDAERR(cudaMalloc(&d_destParams, 2 * sizeof(DestParamsGPU)));
        CHECKCUDAERR(cudaMalloc(&d_expsArgs, sizeof(ExpsArguments)));
    }

    ~StreamData() {
        delete transcript;
        delete transcript_helper;
        CHECKCUDAERR(cudaFree(params));
        CHECKCUDAERR(cudaFree(d_destParams));
        CHECKCUDAERR(cudaFree(d_expsArgs));
    }

    void reset(){
        cudaSetDevice(gpuId);
        cudaEventDestroy(end_event);
        cudaEventCreate(&end_event);
        status = 3;

        extraStream = false;
        streamsUsed = 1;

        root = nullptr;
        pSetupCtx = nullptr;
        proofBuffer = nullptr;
    }

    void free(){
        cudaSetDevice(gpuId);
        cudaStreamDestroy(stream);
        cudaEventDestroy(end_event);
        cudaFreeHost(pinned_buffer);
        cudaFreeHost(pinned_buffer_proof);
        cudaFreeHost(pinned_buffer_exps_params);
        cudaFreeHost(pinned_buffer_exps_args);
        cudaFreeHost(pinned_params);
    }
};
struct DeviceCommitBuffers
{
    gl64_t **d_constPols;
    gl64_t **d_constPolsAggregation;
    gl64_t **d_aux_trace;
    bool recursive;
    uint64_t max_size_proof;

    uint32_t  n_gpus;
    uint32_t* my_gpu_ids;
    uint32_t* gpus_g2l; 
    uint32_t n_total_streams;
    std::mutex mutex_slot_selection;
    StreamData *streamsData;

    std::map<std::pair<uint64_t, uint64_t>, std::map<std::string, std::vector<AirInstanceInfo *>>> air_instances;
};

void copy_to_device_in_chunks(
    DeviceCommitBuffers* d_buffers,
    const void* src,
    void* dst,
    uint64_t total_size,
    uint64_t streamId
    );


void load_and_copy_to_device_in_chunks(
    DeviceCommitBuffers* d_buffers,
    const char* bufferPath,
    void* dst,
    uint64_t total_size,
    uint64_t streamId
    );

#endif


//     inline gl64_t& operator-=(const gl64_t& b)
//     {
//         uint64_t tmp;
//         uint32_t borrow;
//         asm("{ .reg.pred %top;");

// # ifdef GL64_PARTIALLY_REDUCED
//         asm("add.cc.u64 %0, %2, %3; addc.u32 %1, 0, 0;"
//             : "=l"(tmp), "=r"(borrow)
//             : "l"(val), "l"(MOD));
//         asm("setp.eq.u32 %top, %0, 0;" :: "r"(borrow));
//         asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
// # endif

//         // asm("mov.b64 %0, %1;" 
//         //     : "=l"(tmp) 
//         //     : "l"(b.val));

//         // asm("setp.ge.u64 %top, %1, %2;"  // Set predicate if tmp >= MOD
//         //     "@%top sub.u64 %0, %1, %2;"   // If true, subtract MOD from tmp
//         //     : "+l"(tmp)
//         //     : "l"(tmp), "l"(MOD));
            
//         asm("sub.cc.u64 %0, %0, %2; subc.u32 %1, 0, 0;"
//             : "+l"(val), "=r"(borrow)
//             : "l"(b.val));
//         asm("add.u64 %0, %1, %2;" : "=l"(tmp) : "l"(val), "l"(MOD));
//         asm("setp.ne.u32 %top, %0, 0;" :: "r"(borrow));
//         asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
//         asm("}");

//         return *this;
//    }

//     inline gl64_t& operator+=(const gl64_t& b)
//     {
//         from();

//         uint64_t tmp;
//         uint32_t carry;

//         asm("add.cc.u64 %0, %0, %2; addc.u32 %1, 0, 0;"
//             : "+l"(val), "=r"(carry)
//             : "l"(b.val));

//         asm("{ .reg.pred %top;");
// # ifdef GL64_PARTIALLY_REDUCED
//         asm("sub.u64 %0, %1, %2;"
//             : "=l"(tmp)
//             : "l"(val), "l"(MOD));
//         asm("setp.ne.u32 %top, %0, 0;" :: "r"(carry));
//         asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
// # else
//         asm("sub.cc.u64 %0, %2, %3; subc.u32 %1, %1, 0;"
//             : "=l"(tmp), "+r"(carry)
//             : "l"(val), "l"(MOD));
//         asm("setp.eq.u32 %top, %0, 0;" :: "r"(carry));
//         asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
//         // asm("setp.ge.u64 %top, %0, %1;" : : "l"(val), "l"(MOD));
//         // asm("@%top sub.u64 %0, %0, %1;" : "+l"(val) : "l"(MOD));
// # endif
//         asm("}");

//         return *this;
//     }