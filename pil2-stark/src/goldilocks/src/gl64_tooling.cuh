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

    bool stored_tree = false;

    ExpressionsGPU *expressions_gpu;
    int64_t *opening_points;

    uint64_t numBatchesEvals;
    EvalInfo **evalsInfo;
    uint64_t *evalsInfoSizes;

    EvalInfo **evalsInfoFRI;
    uint64_t *evalsInfoFRISizes;
    
    SetupCtx *setupCtx;

    Goldilocks::Element *verkeyRoot;

    bool is_packed = false;
    uint64_t num_packed_words = 0;
    uint64_t *unpack_info = nullptr;

    uint64_t nStreams = 1;

    AirInstanceInfo(uint64_t airgroupId, uint64_t airId, SetupCtx *setupCtx, Goldilocks::Element *verkeyRoot_, PackedInfo *packedInfo, uint64_t nStreams_): setupCtx(setupCtx), airgroupId(airgroupId), airId(airId), nStreams(nStreams_) {
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
                                                                        : 2;
                std::string stage = type == "cm" ? "cm" + to_string(polInfo.stage) : type == "custom" ? setupCtx->starkInfo.customCommits[polInfo.commitId].name + "0" : "const";
                evalsInfoHost[nEvals].stagePos = polInfo.stagePos;
                evalsInfoHost[nEvals].offset = setupCtx->starkInfo.mapOffsets[std::make_pair(stage, true)];
                evalsInfoHost[nEvals].stageCols = setupCtx->starkInfo.mapSectionsN[stage];
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

        EvalInfo **evalsInfoFRI_ = new EvalInfo*[nOpeningPoints];
        uint64_t *evalsInfoFRISizes_ = new uint64_t[nOpeningPoints];

        std::fill(evalsInfoFRISizes_, evalsInfoFRISizes_ + nOpeningPoints, 0);
        for (uint64_t i = 0; i < setupCtx->starkInfo.evMap.size(); i++) {
            evalsInfoFRISizes_[setupCtx->starkInfo.evMap[i].openingPos]++;
        }

        EvalInfo** evalsInfoByOpeningPos = new EvalInfo*[nOpeningPoints];
        for (uint64_t pos = 0; pos < nOpeningPoints; pos++) {
            evalsInfoByOpeningPos[pos] = new EvalInfo[evalsInfoFRISizes_[pos]];
        }

        std::fill(evalsInfoFRISizes_, evalsInfoFRISizes_ + nOpeningPoints, 0);
        for (uint64_t i = 0; i < setupCtx->starkInfo.evMap.size(); i++) {
            EvMap ev = setupCtx->starkInfo.evMap[i];
            uint64_t pos = ev.openingPos;

            std::string type = (ev.type == EvMap::eType::cm) ? "cm" :
                            (ev.type == EvMap::eType::custom) ? "custom" : "fixed";

            PolMap polInfo = (type == "cm")      ? setupCtx->starkInfo.cmPolsMap[ev.id] :
                            (type == "custom")  ? setupCtx->starkInfo.customCommitsMap[ev.commitId][ev.id] :
                                                setupCtx->starkInfo.constPolsMap[ev.id];

            EvalInfo* evInfo = &evalsInfoByOpeningPos[pos][evalsInfoFRISizes_[pos]];
            evInfo->type = (type == "cm") ? 0 : (type == "custom") ? 1 : 2;
            std::string stage = type == "cm" ? "cm" + to_string(polInfo.stage) : type == "custom" ? setupCtx->starkInfo.customCommits[polInfo.commitId].name + "0" : "const";
            evInfo->stagePos = polInfo.stagePos;
            evInfo->offset = setupCtx->starkInfo.mapOffsets[std::make_pair(stage, true)];
            evInfo->stageCols = setupCtx->starkInfo.mapSectionsN[stage];
            evInfo->dim = polInfo.dim;
            evInfo->evalPos = i;
            evInfo->openingPos = pos;

            evalsInfoFRISizes_[pos]++;
        }

        for (uint64_t opening = 0; opening < nOpeningPoints; opening++) {
            CHECKCUDAERR(cudaMalloc(&evalsInfoFRI_[opening], evalsInfoFRISizes_[opening] * sizeof(EvalInfo)));
            CHECKCUDAERR(cudaMemcpy(evalsInfoFRI_[opening], evalsInfoByOpeningPos[opening],
                                    evalsInfoFRISizes_[opening] * sizeof(EvalInfo),
                                    cudaMemcpyHostToDevice));
            delete[] evalsInfoByOpeningPos[opening];
        }
        
        CHECKCUDAERR(cudaMalloc(&evalsInfoFRI, nOpeningPoints * sizeof(EvalInfo*)));
        CHECKCUDAERR(cudaMemcpy(evalsInfoFRI, evalsInfoFRI_, nOpeningPoints * sizeof(EvalInfo*), cudaMemcpyHostToDevice));
        
        delete[] evalsInfoFRI_;
        
        CHECKCUDAERR(cudaMalloc(&evalsInfoFRISizes, nOpeningPoints * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMemcpy(evalsInfoFRISizes, evalsInfoFRISizes_, nOpeningPoints * sizeof(uint64_t), cudaMemcpyHostToDevice));
        
        delete[] evalsInfoFRISizes_;
        delete[] evalsInfoByOpeningPos;

        if (packedInfo != nullptr) {
            is_packed = packedInfo->is_packed;
            num_packed_words = packedInfo->num_packed_words;
            uint64_t nCols = setupCtx->starkInfo.mapSectionsN["cm1"];
            if (is_packed && num_packed_words > 0) {
                CHECKCUDAERR(cudaMalloc(&unpack_info, nCols * sizeof(uint64_t)));
                CHECKCUDAERR(cudaMemcpy(unpack_info, packedInfo->unpack_info, nCols * sizeof(uint64_t), cudaMemcpyHostToDevice));
            }
        }
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

        if (evalsInfoFRI != nullptr) {
            uint64_t nOpeningPoints = setupCtx->starkInfo.openingPoints.size();
            
            EvalInfo **host_evalsInfoFRI = new EvalInfo*[nOpeningPoints];
            CHECKCUDAERR(cudaMemcpy(host_evalsInfoFRI, evalsInfoFRI, nOpeningPoints * sizeof(EvalInfo*), cudaMemcpyDeviceToHost));
            
            for (uint64_t i = 0; i < nOpeningPoints; ++i) {
                if (host_evalsInfoFRI[i] != nullptr) {
                    CHECKCUDAERR(cudaFree(host_evalsInfoFRI[i]));
                }
            }
            
            delete[] host_evalsInfoFRI;
            
            CHECKCUDAERR(cudaFree(evalsInfoFRI));
        }

        if (evalsInfoFRISizes != nullptr) {
            CHECKCUDAERR(cudaFree(evalsInfoFRISizes));
        }

        if (unpack_info != nullptr) {
            CHECKCUDAERR(cudaFree(unpack_info));
        }
    }
};


struct StreamData{

    //const data
    cudaStream_t stream;
    uint32_t gpuId;
    uint64_t localStreamId;
    StepsParams *pinned_params;
    Goldilocks::Element *pinned_buffer_proof;
    Goldilocks::Element *pinned_buffer_exps_params;
    Goldilocks::Element *pinned_buffer_exps_args;

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
    int64_t instanceId;
    string proofType;
        
    bool recursive;
    bool extraStream;
    uint64_t streamsUsed;

    std::mutex mutex_stream_selection;
    
    void initialize(uint64_t max_size_proof, uint32_t gpuId_, uint32_t localStreamId_, bool recursive_){
        uint64_t maxExps = 20000; // TODO: CALCULATE IT PROPERLY!
        cudaSetDevice(gpuId_);
        CHECKCUDAERR(cudaStreamCreate(&stream));
        timer.init(stream);
        gpuId = gpuId_;
        localStreamId = localStreamId_;
        recursive = recursive_;
        cudaEventCreate(&end_event);
        instanceId = -1;
        status = 0;
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

    void reset(bool reset_status){
        cudaSetDevice(gpuId);
        cudaEventDestroy(end_event);
        cudaEventCreate(&end_event);
        status = reset_status ? 0 : 3;

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
    gl64_t ***d_aux_trace;
    gl64_t ***d_aux_traceAggregation;
    Goldilocks::Element **pinned_buffer;
    Goldilocks::Element **pinned_buffer_extra;
    bool recursive;
    uint64_t max_size_proof;

    uint64_t pinned_size = 128 * 1024 * 1024; //256MB

    uint32_t  n_gpus;
    uint32_t* my_gpu_ids;
    uint32_t* gpus_g2l; 
    uint32_t n_total_streams;
    uint32_t n_streams;
    uint32_t n_recursive_streams;
    std::mutex *mutex_pinned;
    StreamData *streamsData;

    std::map<std::pair<uint64_t, uint64_t>, std::map<std::string, std::vector<AirInstanceInfo *>>> air_instances;
};

void copy_to_device_in_chunks(
    DeviceCommitBuffers* d_buffers,
    const void* src,
    void* dst,
    uint64_t total_size,
    uint64_t streamId,
    TimerGPU &timer
    );

void load_and_copy_to_device_in_chunks(
    DeviceCommitBuffers* d_buffers,
    const char* bufferPath,
    void* dst,
    uint64_t total_size,
    uint64_t streamId
    );

#endif