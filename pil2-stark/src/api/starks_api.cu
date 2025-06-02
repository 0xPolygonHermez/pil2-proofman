#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#include "omp.h"
#include "starks_api.hpp"
#include "starks_api_internal.hpp"
#include <cstring>
#include <thread>

#ifdef __USE_CUDA__
#include "gen_recursive_proof.cuh"
#include "gen_proof.cuh"
#include "gen_commit.cuh"
#include "poseidon2_goldilocks.cu"
#include <cuda_runtime.h>
#include <mutex>


struct MaxSizes
{
    uint64_t totalConstPols;
    uint64_t maxAuxTraceArea;
    uint64_t totalConstPolsAggregation;
};

uint32_t selectStream(DeviceCommitBuffers* d_buffers);
void reserveStream(DeviceCommitBuffers* d_buffers, uint32_t streamId);

void closeStreamTimer(TimerGPU &timer, bool isProve);
void get_proof(DeviceCommitBuffers *d_buffers, uint64_t streamId);
void get_commit_root(DeviceCommitBuffers *d_buffers, uint64_t streamId);



void *gen_device_buffers(void *maxSizes_, uint32_t node_rank, uint32_t node_size)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount % node_size != 0) {
        zklog.error("Device count must be divisible by numer of processe per node");
        exit(1);
    }
    if (deviceCount < node_size) {
        zklog.error("Numer of processes per node can not be greater than number of GPUs");
        exit(1);
    }
    MaxSizes *maxSizes = (MaxSizes *)maxSizes_;

    DeviceCommitBuffers *d_buffers = new DeviceCommitBuffers();
    d_buffers->n_gpus = (uint32_t) deviceCount / node_size;
    d_buffers->gpus_g2l = (uint32_t *)malloc(deviceCount * sizeof(uint32_t));
    d_buffers->my_gpu_ids = (uint32_t *)malloc(d_buffers->n_gpus * sizeof(uint32_t));
    for (uint32_t i = 0; i < d_buffers->n_gpus; i++) {
        d_buffers->my_gpu_ids[i] = node_rank * d_buffers->n_gpus + i;
        d_buffers->gpus_g2l[d_buffers->my_gpu_ids[i]] = i;
    }
    d_buffers->d_aux_trace = (gl64_gpu **)malloc(deviceCount * sizeof(gl64_gpu*));
    d_buffers->d_constPols = (gl64_gpu **)malloc(deviceCount * sizeof(gl64_gpu*));
    d_buffers->d_constPolsAggregation = (gl64_gpu **)malloc(deviceCount * sizeof(gl64_gpu*));

    for (int i = 0; i < d_buffers->n_gpus; i++) {
        cudaSetDevice(d_buffers->my_gpu_ids[i]);
        CHECKCUDAERR(cudaMalloc(&d_buffers->d_aux_trace[i], maxSizes->maxAuxTraceArea * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMalloc(&d_buffers->d_constPols[i], maxSizes->totalConstPols * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMalloc(&d_buffers->d_constPolsAggregation[i], maxSizes->totalConstPolsAggregation * sizeof(Goldilocks::Element)));
    }
    init_gpu_const_2(d_buffers->my_gpu_ids, d_buffers->n_gpus);

    TranscriptGL_GPU::init_const(d_buffers->my_gpu_ids, d_buffers->n_gpus);
    return (void *)d_buffers;
}

uint64_t gen_device_streams(void *d_buffers_, uint64_t maxSizeTrace, uint64_t maxSizeContribution, uint64_t maxSizeProverBuffer, uint64_t maxSizeConst, uint64_t maxSizeConstTree, uint64_t maxSizeTraceAggregation, uint64_t maxSizeProverBufferAggregation, uint64_t maxSizeConstAggregation, uint64_t maxSizeConstTreeAggregation, uint64_t maxProofSize, uint64_t maxProofsPerGPU) {
    
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    d_buffers->max_size_prover_buffer = maxSizeProverBuffer;
    d_buffers->max_size_trace = maxSizeTrace;
    d_buffers->max_size_contribution = maxSizeContribution;
    d_buffers->max_size_const = maxSizeConst;
    d_buffers->max_size_const_tree = maxSizeConstTree;
    d_buffers->max_size_trace_aggregation = maxSizeTraceAggregation;
    d_buffers->max_size_prover_buffer_aggregation = maxSizeProverBufferAggregation;
    d_buffers->max_size_const_aggregation = maxSizeConstAggregation;
    d_buffers->max_size_const_tree_aggregation = maxSizeConstTreeAggregation;
    d_buffers->max_size_proof = maxProofSize;
    d_buffers->n_streams = d_buffers->n_gpus * maxProofsPerGPU;
    d_buffers->n_streams_per_gpu = maxProofsPerGPU;

    if (d_buffers->streamsData != nullptr) {
        for (uint64_t i = 0; i < d_buffers->n_streams; i++) {
            d_buffers->streamsData[i].free();
        }
        delete[] d_buffers->streamsData;
    }
    d_buffers->streamsData = new StreamData[d_buffers->n_streams];

    for(uint64_t i=0; i< d_buffers->n_gpus; ++i){
        for (uint64_t j = 0; j < maxProofsPerGPU; j++) {
            d_buffers->streamsData[i*maxProofsPerGPU+j].initialize(maxSizeTrace, maxProofSize, maxSizeConst, maxSizeConstAggregation, maxSizeConstTree, maxSizeConstTreeAggregation, d_buffers->my_gpu_ids[i], j);
        }
    }

    return d_buffers->n_gpus;
}

void free_device_buffers(void *d_buffers_)
{
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;

    for (int i = 0; i < d_buffers->n_gpus; ++i) {
        cudaSetDevice(d_buffers->my_gpu_ids[i]);
        CHECKCUDAERR(cudaFree(d_buffers->d_aux_trace[i]));
        CHECKCUDAERR(cudaFree(d_buffers->d_constPols[i]));
        CHECKCUDAERR(cudaFree(d_buffers->d_constPolsAggregation[i]));
    }
    free(d_buffers->d_aux_trace);
    free(d_buffers->d_constPols);
    free(d_buffers->d_constPolsAggregation);
    
    if (d_buffers->streamsData != nullptr) {
        for (uint64_t i = 0; i < d_buffers->n_streams; i++) {
            d_buffers->streamsData[i].free();
        }
        delete[] d_buffers->streamsData;
    }

    for (auto &outer_pair : d_buffers->air_instances) {
        for (auto &inner_pair : outer_pair.second) {
            for (AirInstanceInfo *ptr : inner_pair.second) {
                delete ptr;
            }
        }
    }

    delete d_buffers;
}


void load_device_setup(uint64_t airgroupId, uint64_t airId, char *proofType, void *pSetupCtx_, void *d_buffers_, void *verkeyRoot_) {
    
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    Goldilocks::Element *verkeyRoot = (Goldilocks::Element *)verkeyRoot_;

    std::pair<uint64_t, uint64_t> key = {airgroupId, airId};

    if (d_buffers->air_instances[key][proofType].empty()) {
        d_buffers->air_instances[key][proofType].resize(d_buffers->n_gpus, nullptr);
    }

    for(int i=0; i<d_buffers->n_gpus; ++i){
        cudaSetDevice(d_buffers->my_gpu_ids[i]);
        d_buffers->air_instances[key][proofType][i] = new AirInstanceInfo(airgroupId, airId, setupCtx, verkeyRoot);
    }
}

void load_device_const_pols(uint64_t airgroupId, uint64_t airId, uint64_t initial_offset, void *d_buffers_, char *constFilename, uint64_t constSize, char *constTreeFilename, uint64_t constTreeSize, char *proofType) {
    
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint64_t sizeConstPols = constSize * sizeof(Goldilocks::Element);
    uint64_t sizeConstTree = constTreeSize * sizeof(Goldilocks::Element);
    
    std::pair<uint64_t, uint64_t> key = {airgroupId, airId};

    uint64_t const_pols_offset = initial_offset;
    uint64_t const_tree_offset = initial_offset + constSize;

    Goldilocks::Element *constPols = new Goldilocks::Element[constSize];
    Goldilocks::Element *constTree = new Goldilocks::Element[constTreeSize];

    loadFileParallel(constPols, constFilename, sizeConstPols);
    loadFileParallel(constTree, constTreeFilename, sizeConstTree);
    
    for(int i=0; i<d_buffers->n_gpus; ++i){
        cudaSetDevice(d_buffers->my_gpu_ids[i]);
        gl64_gpu *d_constPols = (strcmp(proofType, "basic") == 0) ? d_buffers->d_constPols[i] : d_buffers->d_constPolsAggregation[i];
        gl64_gpu *d_constTree = (strcmp(proofType, "basic") == 0) ? d_buffers->d_constPols[i] : d_buffers->d_constPolsAggregation[i];
        CHECKCUDAERR(cudaMemcpy(d_constPols + const_pols_offset, constPols, sizeConstPols, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(d_constTree + const_tree_offset, constTree, sizeConstTree, cudaMemcpyHostToDevice));
        AirInstanceInfo* air_instance_info = d_buffers->air_instances[key][proofType][i];
        air_instance_info->const_pols_offset = const_pols_offset;
        air_instance_info->const_tree_offset = const_tree_offset;
        air_instance_info->stored = true;
    }

    delete[] constPols;
    delete[] constTree;
}

uint64_t gen_proof(void *pSetupCtx_, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params_, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers_, bool skipRecalculation, uint64_t streamId_, char *constPolsPath,  char *constTreePath) {

    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint32_t streamId = skipRecalculation ? streamId_ : selectStream(d_buffers);
    if (skipRecalculation) reserveStream(d_buffers, streamId);
    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];
    uint64_t slotId = d_buffers->streamsData[streamId].slotId;
    set_device(gpuId);

    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    StepsParams *params = (StepsParams *)params_;
    cudaStream_t stream = d_buffers->streamsData[streamId].stream;
    TimerGPU &timer = d_buffers->streamsData[streamId].timer;

    gl64_gpu *d_aux_trace = (gl64_gpu *)d_buffers->d_aux_trace[gpuLocalId] + slotId*d_buffers->max_size_prover_buffer;

    uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
    uint64_t nCols = setupCtx->starkInfo.mapSectionsN["cm1"];
    uint64_t sizeTrace = N * (setupCtx->starkInfo.mapSectionsN["cm1"]) * sizeof(Goldilocks::Element);
    uint64_t sizeConstPols = N * (setupCtx->starkInfo.nConstants) * sizeof(Goldilocks::Element);
    uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);
  
    auto key = std::make_pair(airgroupId, airId);
    std::string proofType = "basic";
    AirInstanceInfo *air_instance_info = d_buffers->air_instances[key][proofType][gpuLocalId];
    uint64_t offset = 0;
    
    if (setupCtx->starkInfo.mapTotalNCustomCommitsFixed > 0) {
        Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx->starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
        CHECKCUDAERR(cudaMemcpyAsync(pCustomCommitsFixed, params->pCustomCommitsFixed, setupCtx->starkInfo.mapTotalNCustomCommitsFixed * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
    }

    if (!air_instance_info->stored && (d_buffers->streamsData[streamId].airgroupId != airgroupId || d_buffers->streamsData[streamId].airId != airId || d_buffers->streamsData[streamId].proofType != "basic")) {
        loadFileParallel(d_buffers->streamsData[streamId].pinned_buffer_const, constPolsPath, sizeConstPols);
        loadFileParallel(d_buffers->streamsData[streamId].pinned_buffer_const_tree, constTreePath, sizeConstTree);
    }

    d_buffers->streamsData[streamId].pSetupCtx = pSetupCtx_;
    d_buffers->streamsData[streamId].proofBuffer = proofBuffer;
    d_buffers->streamsData[streamId].proofFile = string(proofFile);
    d_buffers->streamsData[streamId].airgroupId = airgroupId;
    d_buffers->streamsData[streamId].airId = airId;
    d_buffers->streamsData[streamId].instanceId = instanceId;
    d_buffers->streamsData[streamId].proofType = "basic";

    if (!skipRecalculation) {
        memcpy(d_buffers->streamsData[streamId].pinned_buffer, (Goldilocks::Element *)params->trace, N * nCols * sizeof(Goldilocks::Element));
    }
    offset = N * nCols;
    memcpy(&d_buffers->streamsData[streamId].pinned_buffer[offset], params->publicInputs, setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element));
    offset += setupCtx->starkInfo.nPublics;
    memcpy(&d_buffers->streamsData[streamId].pinned_buffer[offset], params->proofValues, setupCtx->starkInfo.proofValuesSize * sizeof(Goldilocks::Element));
    offset += setupCtx->starkInfo.proofValuesSize;
    memcpy(&d_buffers->streamsData[streamId].pinned_buffer[offset], params->airgroupValues, setupCtx->starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element));
    offset += setupCtx->starkInfo.airgroupValuesSize;
    memcpy(&d_buffers->streamsData[streamId].pinned_buffer[offset], params->airValues, setupCtx->starkInfo.airValuesSize * sizeof(Goldilocks::Element));
    offset += setupCtx->starkInfo.airValuesSize;
    memcpy(&d_buffers->streamsData[streamId].pinned_buffer[offset], globalChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element));

    uint64_t offsetStage1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];
    uint64_t offsetAirgroupValues = setupCtx->starkInfo.mapOffsets[std::make_pair("airgroupvalues", false)];
    uint64_t offsetAirValues = setupCtx->starkInfo.mapOffsets[std::make_pair("airvalues", false)];
    uint64_t offsetProofValues = setupCtx->starkInfo.mapOffsets[std::make_pair("proofvalues", false)];
    uint64_t offsetChallenge = setupCtx->starkInfo.mapOffsets[std::make_pair("challenge", false)];

    offset = 0;
    if (!skipRecalculation) {
        CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetStage1, &d_buffers->streamsData[streamId].pinned_buffer[offset], N * nCols * sizeof(Goldilocks::Element ), cudaMemcpyHostToDevice, stream));
    }
    offset += N * nCols;
    CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetPublicInputs, &d_buffers->streamsData[streamId].pinned_buffer[offset], setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
    offset += setupCtx->starkInfo.nPublics;
    if (setupCtx->starkInfo.proofValuesSize > 0) {
        CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetProofValues, &d_buffers->streamsData[streamId].pinned_buffer[offset], setupCtx->starkInfo.proofValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
        offset += setupCtx->starkInfo.proofValuesSize;
    }
    if (setupCtx->starkInfo.airgroupValuesSize > 0) {
        CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetAirgroupValues, &d_buffers->streamsData[streamId].pinned_buffer[offset], setupCtx->starkInfo.airgroupValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
        offset += setupCtx->starkInfo.airgroupValuesSize;
    }
    if (setupCtx->starkInfo.airValuesSize > 0) {
        CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetAirValues, &d_buffers->streamsData[streamId].pinned_buffer[offset], setupCtx->starkInfo.airValuesSize * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
        offset += setupCtx->starkInfo.airValuesSize;
    }
    CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetChallenge, &d_buffers->streamsData[streamId].pinned_buffer[offset], FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
    offset += FIELD_EXTENSION;

    gl64_gpu *d_const_pols;
    gl64_gpu *d_const_tree;
    if (air_instance_info->stored) {
        d_const_pols = d_buffers->d_constPols[gpuLocalId] + air_instance_info->const_pols_offset;
        d_const_tree = d_buffers->d_constPols[gpuLocalId] + air_instance_info->const_tree_offset;
    } else {
        uint64_t offsetConstTree = setupCtx->starkInfo.mapOffsets[std::make_pair("const", true)];
        uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
        CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetConstPols, d_buffers->streamsData[streamId].pinned_buffer_const, sizeConstPols, cudaMemcpyHostToDevice, stream));
        CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetConstTree, d_buffers->streamsData[streamId].pinned_buffer_const_tree, sizeConstTree, cudaMemcpyHostToDevice, stream));
        d_const_pols = d_aux_trace + offsetConstPols;
        d_const_tree = d_aux_trace + offsetConstTree;
    }


    genProof_gpu(*setupCtx, d_aux_trace, d_const_pols, d_const_tree, streamId, d_buffers, air_instance_info, skipRecalculation, timer, stream);
    cudaEventRecord(d_buffers->streamsData[streamId].end_event, stream);
    d_buffers->streamsData[streamId].status = 2;
    return streamId;


}

void get_proof(DeviceCommitBuffers *d_buffers, uint64_t streamId) {

    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    set_device(gpuId);

    SetupCtx *setupCtx = (SetupCtx*) d_buffers->streamsData[streamId].pSetupCtx;
    uint64_t airgroupId = d_buffers->streamsData[streamId].airgroupId;
    uint64_t airId = d_buffers->streamsData[streamId].airId;
    uint64_t instanceId = d_buffers->streamsData[streamId].instanceId;
    uint64_t * proofBuffer = d_buffers->streamsData[streamId].proofBuffer;
    string proofType = d_buffers->streamsData[streamId].proofType;
    string proofFile = d_buffers->streamsData[streamId].proofFile;
    TimerGPU &timer = d_buffers->streamsData[streamId].timer;

    closeStreamTimer(timer, true);

    writeProof(*setupCtx, d_buffers->streamsData[streamId].pinned_buffer_proof, proofBuffer, airgroupId, airId, instanceId, proofFile);

    if (proof_done_callback != nullptr) {
        proof_done_callback(instanceId, proofType.c_str());
    }
}

void get_stream_proofs(void *d_buffers_){
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    std::lock_guard<std::mutex> lock(d_buffers->mutex_slot_selection);
    for (uint64_t i = 0; i < d_buffers->n_streams; i++) {
        if (d_buffers->streamsData[i].status == 0 || d_buffers->streamsData[i].status == 3) continue;
        set_device(d_buffers->streamsData[i].gpuId);
        CHECKCUDAERR(cudaStreamSynchronize(d_buffers->streamsData[i].stream));
        if(d_buffers->streamsData[i].root != nullptr) {
            get_commit_root(d_buffers, i);
        }else{
            get_proof(d_buffers, i);
        }
        d_buffers->streamsData[i].reset();        
    }
}

void get_stream_proofs_non_blocking(void *d_buffers_){
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    std::lock_guard<std::mutex> lock(d_buffers->mutex_slot_selection);
    for (uint64_t i = 0; i < d_buffers->n_streams; i++) {
        if(d_buffers->streamsData[i].status==2 &&  cudaEventQuery(d_buffers->streamsData[i].end_event) == cudaSuccess){
            set_device(d_buffers->streamsData[i].gpuId);
            if(d_buffers->streamsData[i].root != nullptr) {
                get_commit_root(d_buffers, i);
            }else{
                get_proof(d_buffers, i);
            }
            d_buffers->streamsData[i].reset();        
        }
    }
}

void get_stream_id_proof(void *d_buffers_, uint64_t streamId) {
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    set_device(d_buffers->streamsData[streamId].gpuId);
    CHECKCUDAERR(cudaStreamSynchronize(d_buffers->streamsData[streamId].stream));
    if(d_buffers->streamsData[streamId].root != nullptr) {
            get_commit_root(d_buffers, streamId);
        }else{
            get_proof(d_buffers, streamId);
        }
    d_buffers->streamsData[streamId].reset();
}

uint64_t gen_recursive_proof(void *pSetupCtx_, char *globalInfoFile, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *trace, void *aux_trace, void *pConstPols, void *pConstTree, void *pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers_, char *constPolsPath, char *constTreePath, char *proofType)
{
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint32_t streamId = selectStream(d_buffers);
    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];
    uint64_t slotId =  d_buffers->streamsData[streamId].slotId;
    set_device(gpuId);

    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    cudaStream_t stream = d_buffers->streamsData[streamId].stream;
    TimerGPU &timer = d_buffers->streamsData[streamId].timer;
    
    gl64_gpu *d_trace = (gl64_gpu *)d_buffers->d_aux_trace[gpuLocalId] + slotId*d_buffers->max_size_prover_buffer;
    gl64_gpu *d_aux_trace = d_trace + d_buffers->max_size_trace_aggregation;

    uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
    uint64_t nCols = setupCtx->starkInfo.mapSectionsN["cm1"];
    uint64_t sizeTrace = N * (setupCtx->starkInfo.mapSectionsN["cm1"]) * sizeof(Goldilocks::Element);
    uint64_t sizeConstPols = N * (setupCtx->starkInfo.nConstants) * sizeof(Goldilocks::Element);
    uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);

    auto key = std::make_pair(airgroupId, airId);
    AirInstanceInfo *air_instance_info = d_buffers->air_instances[key][string(proofType)][gpuLocalId];

    memcpy(d_buffers->streamsData[streamId].pinned_buffer, (Goldilocks::Element *)trace, N * nCols * sizeof(Goldilocks::Element));
    if (!air_instance_info->stored && (d_buffers->streamsData[streamId].airgroupId != airgroupId || d_buffers->streamsData[streamId].airId != airId || d_buffers->streamsData[streamId].proofType != string(proofType))) {
        loadFileParallel(d_buffers->streamsData[streamId].pinned_buffer_const, constPolsPath, sizeConstPols);
        loadFileParallel(d_buffers->streamsData[streamId].pinned_buffer_const_tree, constTreePath, sizeConstTree);
    }

    d_buffers->streamsData[streamId].pSetupCtx = pSetupCtx_;
    d_buffers->streamsData[streamId].proofBuffer = proofBuffer;
    d_buffers->streamsData[streamId].proofFile = string(proof_file);
    d_buffers->streamsData[streamId].airgroupId = airgroupId;
    d_buffers->streamsData[streamId].airId = airId;
    d_buffers->streamsData[streamId].instanceId = instanceId;
    d_buffers->streamsData[streamId].proofType = string(proofType);

    memcpy(&d_buffers->streamsData[streamId].pinned_buffer[N * nCols], (Goldilocks::Element *)pPublicInputs, setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element));

    uint64_t offsetPublicInputs = setupCtx->starkInfo.mapOffsets[std::make_pair("publics", false)];
    CHECKCUDAERR(cudaMemcpyAsync(d_trace, d_buffers->streamsData[streamId].pinned_buffer, sizeTrace, cudaMemcpyHostToDevice, stream));
    CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetPublicInputs, &d_buffers->streamsData[streamId].pinned_buffer[N * nCols], setupCtx->starkInfo.nPublics * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice, stream));
    
    gl64_gpu *d_const_pols;
    gl64_gpu *d_const_tree;
    if (air_instance_info->stored) {
        d_const_pols = d_buffers->d_constPolsAggregation[gpuLocalId] + air_instance_info->const_pols_offset;
        d_const_tree = d_buffers->d_constPolsAggregation[gpuLocalId] + air_instance_info->const_tree_offset;
    } else {
        uint64_t offsetConstTree = setupCtx->starkInfo.mapOffsets[std::make_pair("const", true)];
        uint64_t offsetConstPols = setupCtx->starkInfo.mapOffsets[std::make_pair("const", false)];
        d_const_pols = d_aux_trace + offsetConstPols;
        d_const_tree = d_aux_trace + offsetConstTree;
        CHECKCUDAERR(cudaMemcpyAsync(d_const_pols, d_buffers->streamsData[streamId].pinned_buffer_const, sizeConstPols, cudaMemcpyHostToDevice, stream));
        CHECKCUDAERR(cudaMemcpyAsync(d_const_tree, d_buffers->streamsData[streamId].pinned_buffer_const_tree, sizeConstTree, cudaMemcpyHostToDevice, stream));
    }

    genRecursiveProof_gpu<Goldilocks::Element>(*setupCtx, d_trace, d_aux_trace, d_const_pols, d_const_tree, streamId, d_buffers, air_instance_info, instanceId, timer, stream);
    cudaEventRecord(d_buffers->streamsData[streamId].end_event, stream);
    d_buffers->streamsData[streamId].status = 2;
    return streamId;
}

uint64_t commit_witness(uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, uint64_t instanceId, void *root, void *trace, void *auxTrace, void *d_buffers_, void *pSetupCtx_) {

    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx_;
    DeviceCommitBuffers *d_buffers = (DeviceCommitBuffers *)d_buffers_;
    uint32_t streamId = selectStream(d_buffers);
    uint32_t gpuId = d_buffers->streamsData[streamId].gpuId;
    uint32_t gpuLocalId = d_buffers->gpus_g2l[gpuId];
    uint64_t slotId = d_buffers->streamsData[streamId].slotId;
    set_device(gpuId);

    d_buffers->streamsData[streamId].root = root;
    d_buffers->streamsData[streamId].instanceId = instanceId;

    uint64_t N = 1 << nBits;

    cudaStream_t stream = d_buffers->streamsData[streamId].stream;
    TimerGPU &timer = d_buffers->streamsData[streamId].timer;

    gl64_gpu *d_aux_trace = (gl64_gpu *)d_buffers->d_aux_trace[gpuLocalId] + slotId*d_buffers->max_size_prover_buffer;
    uint64_t sizeTrace = N * nCols * sizeof(Goldilocks::Element);
    uint64_t offsetStage1 = setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)];

    memcpy(d_buffers->streamsData[streamId].pinned_buffer, (Goldilocks::Element *)trace, N * nCols * sizeof(Goldilocks::Element));
    CHECKCUDAERR(cudaMemcpyAsync(d_aux_trace + offsetStage1, d_buffers->streamsData[streamId].pinned_buffer, sizeTrace, cudaMemcpyHostToDevice, stream));
    genCommit_gpu(arity, nBits, nBitsExt, nCols, d_aux_trace, d_buffers->streamsData[streamId].pinned_buffer_proof, setupCtx, timer, stream);

    cudaEventRecord(d_buffers->streamsData[streamId].end_event, stream);
    d_buffers->streamsData[streamId].status = 2;
    return streamId;
}

void get_commit_root(DeviceCommitBuffers *d_buffers, uint64_t streamId) {

    set_device(d_buffers->streamsData[streamId].gpuId);
    Goldilocks::Element *root = (Goldilocks::Element *)d_buffers->streamsData[streamId].root;
    memcpy((Goldilocks::Element *)root, d_buffers->streamsData[streamId].pinned_buffer_proof, HASH_SIZE * sizeof(uint64_t));
    closeStreamTimer(d_buffers->streamsData[streamId].timer, false);
    
    uint64_t instanceId = d_buffers->streamsData[streamId].instanceId;

    if (proof_done_callback != nullptr) {
        proof_done_callback(instanceId, "");
    }

}

uint64_t check_device_memory() {
    
    set_device(0); //We assume that all the GPUs have the same characteristics, we only check the GPU 0
    uint64_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    std::cout << "Free memory: " << freeMem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Total memory: " << totalMem / (1024.0 * 1024.0) << " MB" << std::endl;

    return freeMem;
}

// Function to set the CUDA device based on the MPI rank
void set_device_mpi(uint32_t mpi_node_rank){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        exit(1);
    }
    int device = mpi_node_rank % deviceCount;
    cudaSetDevice(device);
}

void set_device(uint32_t gpuId){
    cudaSetDevice(gpuId);
}

uint32_t selectStream(DeviceCommitBuffers* d_buffers){

    std::lock_guard<std::mutex> lock(d_buffers->mutex_slot_selection);
    uint32_t countFreeStreamsGPU[d_buffers->n_gpus];
    uint32_t countUnusedStreams[d_buffers->n_gpus];
    int streamIdxGPU[d_buffers->n_gpus];
    
    for( uint32_t i = 0; i < d_buffers->n_gpus; i++){
        countUnusedStreams[i] = 0;
        countFreeStreamsGPU[i] = 0;
        streamIdxGPU[i] = -1;
    }

    bool someFree = false;
    while (!someFree){
        for (uint32_t i = 0; i < d_buffers->n_streams; i++) {
            if (d_buffers->streamsData[i].status==0 || d_buffers->streamsData[i].status==3 || (d_buffers->streamsData[i].status==2 &&  cudaEventQuery(d_buffers->streamsData[i].end_event) == cudaSuccess)) {

                countFreeStreamsGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]]++;
                if(d_buffers->streamsData[i].status==0){
                    countUnusedStreams[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]]++;
                    streamIdxGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]] = i;
                }
                if( streamIdxGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]] == -1 ){
                    streamIdxGPU[d_buffers->gpus_g2l[d_buffers->streamsData[i].gpuId]] = i;
                }
                someFree = true;
            }
        }
        if (!someFree)
            std::this_thread::sleep_for(std::chrono::microseconds(300)); 
    }

    uint32_t maxFree = 0;
    uint32_t streamId = 0;
    for (uint32_t i = 0; i < d_buffers->n_gpus; i++) {
        if (countFreeStreamsGPU[i] > maxFree || (countFreeStreamsGPU[i] == maxFree && countUnusedStreams[i] > countUnusedStreams[streamId])) {
            maxFree = countFreeStreamsGPU[i];
            streamId = streamIdxGPU[i];
        }
    }

    reserveStream(d_buffers, streamId);
    return streamId;
}

void reserveStream(DeviceCommitBuffers* d_buffers, uint32_t streamId){
    if(d_buffers->streamsData[streamId].status==2 &&  cudaEventQuery(d_buffers->streamsData[streamId].end_event) == cudaSuccess) {

        if(d_buffers->streamsData[streamId].root != nullptr) {
            get_commit_root(d_buffers, streamId);
        }else{
            get_proof(d_buffers, streamId);
        }
        d_buffers->streamsData[streamId].reset();
    }

    d_buffers->streamsData[streamId].status = 1;
}

void closeStreamTimer(TimerGPU &timer, bool isProve) {
    TimerSyncAndLogAllGPU(timer); 
    TimerSyncCategoriesGPU(timer);
    if(isProve)
        TimerLogCategoryContributionsGPU(timer, STARK_GPU_PROOF);
    else
        TimerLogCategoryContributionsGPU(timer, STARK_GPU_COMMIT);
    TimerResetGPU(timer);
}
#endif