#ifndef __GL64_GPU_CUH__
#define __GL64_GPU_CUH__

#include <cstdint>
#include <cassert>
#include <atomic>
#ifdef USE_CUDA_GRAPH
#include <memory>
#include "cuda_graph_cache.cuh"
#endif
#include "goldilocks_base_field.hpp"
#include "goldilocks_trace_layout.cuh"  // Layout enum, getBufferOffset (fromRowMajorToColMajor)
#ifndef __GOLDILOCKS_ENV__
#include "gpu_timer.cuh"
#include <mutex>
#include "cuda_utils.cuh"
#include "transcriptGL.cuh"
#include "expressions_gpu.cuh"
#include <limits.h>
#include "fr.hpp"
#endif
#include "gl64_t.cuh"

// Reduce a Goldilocks value from partially reduced form [0, 2*MOD) to canonical form [0, MOD)
// This is needed when converting Goldilocks values to other field representations (e.g., BN128)
__device__ __forceinline__ uint64_t gl64_reduce(uint64_t val) {
    return (val >= GOLDILOCKS_PRIME) ? (val - GOLDILOCKS_PRIME) : val;
}

// Overload for Goldilocks::Element
__device__ __forceinline__ uint64_t gl64_reduce(const Goldilocks::Element& gl) {
    return gl64_reduce(gl.fe);
}

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

#ifndef __GOLDILOCKS_ENV__
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
    uint64_t* d_num_packed_words;

    // Indexed (compact) cm1 unpack. The program-independent descriptor arrives via PackedInfo
    // at setup; the instruction table is uploaded later, per program, via
    // set_instruction_table(). d_col_source == nullptr is what selects the plain unpack
    // path -- when it is set, d_instr_table must have been registered before the first
    // unpack, and unpack_trace aborts rather than silently falling back to the plain walk.
    uint8_t  *d_col_source = nullptr;   // per column: 0 = row stream, 1 = table stream
    uint64_t  index_bits = 0;           // width of the leading index header in a compact row
    uint64_t  words_per_entry = 0;      // u64 words per instruction-table entry
    uint64_t *d_instr_table = nullptr;  // num_entries * words_per_entry, uploaded per program
    uint64_t  num_entries = 0;

    // Upload (replacing any previous) the program-specific instruction table. Caller must
    // have selected the target GPU. Safe to call repeatedly ACROSS programs, but never
    // while work using d_instr_table is in flight on this GPU -- the cudaFree below would
    // pull the table out from under a running unpack. `words` also refreshes
    // words_per_entry (seeded from PackedInfo at setup); the live program wins.
    void set_instruction_table(const uint64_t *table, uint64_t entries, uint64_t words) {
        if (d_instr_table != nullptr) {
            CHECKCUDAERR(cudaFree(d_instr_table));
            d_instr_table = nullptr;
        }
        num_entries = entries;
        words_per_entry = words;
        if (entries > 0 && words > 0) {
            CHECKCUDAERR(cudaMalloc(&d_instr_table, entries * words * sizeof(uint64_t)));
            CHECKCUDAERR(cudaMemcpy(d_instr_table, table, entries * words * sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
    }

    AirInstanceInfo(uint64_t airgroupId, uint64_t airId, SetupCtx *setupCtx, Goldilocks::Element *verkeyRoot_, PackedInfo *packedInfo): setupCtx(setupCtx), airgroupId(airgroupId), airId(airId) {
        int64_t *d_openingPoints;
        CHECKCUDAERR(cudaMalloc(&d_openingPoints, setupCtx->starkInfo.openingPoints.size() * sizeof(int64_t)));
        CHECKCUDAERR(cudaMemcpy(d_openingPoints, setupCtx->starkInfo.openingPoints.data(), setupCtx->starkInfo.openingPoints.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        opening_points = d_openingPoints;
        expressions_gpu = new ExpressionsGPU(*setupCtx, setupCtx->starkInfo.nrowsPack, setupCtx->starkInfo.maxNBlocks);

        if(verkeyRoot_ == nullptr) {
            verkeyRoot = nullptr;
        }
        else {
            Goldilocks::Element *d_verkeyRoot;
            CHECKCUDAERR(cudaMalloc(&d_verkeyRoot, HASH_SIZE * sizeof(Goldilocks::Element)));
            CHECKCUDAERR(cudaMemcpy(d_verkeyRoot, verkeyRoot_, HASH_SIZE * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
            verkeyRoot = d_verkeyRoot;
        }

        CHECKCUDAERR(cudaMalloc(&d_num_packed_words, sizeof(uint64_t)));


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

        // mapOffsets is a std::map, so a missing key silently reads back as offset 0 --
        // calculateFRIExpression would scribble the head of the arena rather than fail.
        // Checked here because that call site sits inside a cudagraph capture region.
        // Only for proof-generating setups: the verify and verify-constraints branches of
        // StarkInfo::load lay out their own arena and never reach FRI, so the region is
        // legitimately absent there.
        if (!setupCtx->starkInfo.verify_constraints && !setupCtx->starkInfo.verify &&
            setupCtx->starkInfo.mapOffsets.count(std::make_pair("fri_folded", false)) == 0) {
            throw std::runtime_error("AirInstanceInfo: aux_trace has no fri_folded region (StarkInfo not loaded for gpu)");
        }

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
            cudaMemcpy(d_num_packed_words, &num_packed_words, sizeof(uint64_t), cudaMemcpyHostToDevice);

            // Indexed variant descriptor; the table itself arrives via set_instruction_table().
            if (packedInfo->col_source != nullptr) {
                index_bits = packedInfo->index_bits;
                words_per_entry = packedInfo->words_per_entry;
                CHECKCUDAERR(cudaMalloc(&d_col_source, nCols * sizeof(uint8_t)));
                CHECKCUDAERR(cudaMemcpy(d_col_source, packedInfo->col_source, nCols * sizeof(uint8_t), cudaMemcpyHostToDevice));
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
        CHECKCUDAERR(cudaFree(d_num_packed_words));

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

        if (d_col_source != nullptr) {
            CHECKCUDAERR(cudaFree(d_col_source));
        }

        if (d_instr_table != nullptr) {
            CHECKCUDAERR(cudaFree(d_instr_table));
        }
    }
};


// Upper bound on per-stream staged aux_values; call sites assert the actual size fits.
#define PINNED_AUX_VALUES_MAX 65536

// Slot capacity (one slot per expression launch) of the pinned_buffer_exps_* staging
// buffers; stageExpsSlot (expressions_gpu.cu) bounds countId against it. Shared by the
// main streams and the recursiveF buffer so the single bound matches every allocation.
#define PINNED_EXPS_SLOTS 40000

struct StreamData{

    //const data
    cudaStream_t stream;
    uint32_t gpuId;
    uint64_t localStreamId;
    StepsParams *pinned_params;
    Goldilocks::Element *pinned_buffer_proof;
    Goldilocks::Element *pinned_buffer_exps_params;
    Goldilocks::Element *pinned_buffer_exps_args;
    // Per-stream pinned staging for the contributions aux_values H2D, enabling an
    // async copy (no per-copy stream sync); reused only on event-gated stream
    // reselect. Used by commit_witness_gpu only.
    Goldilocks::Element *pinned_aux_values;

    //runtime data
    // Atomic: status is read (unlocked) by wait_trace_h2d_done / callbacks while
    // reserveStream/gen_proof write it, so a plain int would be a data race. The
    // atomic only removes UB on the individual load/store; the check-then-act
    // sequences in selectStream/reserveStream/get_stream_proofs_* are made correct
    // by holding mutex_stream_selection, not by the atomic.
    std::atomic<uint32_t> status{0}; //0: unused, 1: loading, 2: full, 3: reusable (not unused)
    cudaEvent_t end_event;
    // Marks the point where this stream stops reading the caller's host trace buffer, i.e. the
    // trace H2D. Distinct from end_event (the whole commit): the buffer can be recycled as soon
    // as the copy is done, and gating that on the LDE/Merkle work kept the pool starved.
    cudaEvent_t trace_copy_event;
    TimerGPU timer;

    TranscriptGL_GPU *transcript;
    TranscriptGL_GPU *transcript_helper;

    StepsParams *params;
    ExpsArguments *d_expsArgs;
    DestParamsGPU *d_destParams;

    // Disambiguates recurser setups (all share (0,0,"recursive2")) in the recursive-path
    // const-reuse check; empty for normal recursion. Cleared by invalidateContext().
    string recurserId;

    // Scalar "resident witness" marker read locklessly by get_instances_ready (reading
    // proofType there would race concurrent std::string writes). Set by commit_witness,
    // cleared by every proof path and invalidateContext; survives reset() like proofType.
    bool witnessResident;

    //callback inputs
    void *root;
    void *pSetupCtx;
    uint64_t *proofBuffer; 
    string proofFile;
    uint64_t airgroupId; 
    uint64_t airId; 
    int64_t instanceId;
    string proofType;
    uint64_t arity;

    // Which fixed columns this stream's aux trace holds, not which air it last served: airs
    // with identical fixed share a const-pols slot, so the offset is the key. constAggBuffer
    // separates the two const buffers (unrelated offsets), constRecurserId the recursers
    // (one shared slot). custom_fixed is per-air and stays keyed on the air.
    uint64_t constPolsOffset = UINT64_MAX; // UINT64_MAX = nothing cached
    // Where the unpacked pols land in the aux trace. Part of the key because two airs can
    // share a slot yet lay out ("const", false) differently -- a preallocated const tree
    // moves it, and constPolsAliasTree moves it into the tree's node area.
    uint64_t constAuxOffset = 0;
    bool constAggBuffer = false;
    string constRecurserId;
    // ("const", true) holds this slot's tree. Separate from the pols because the
    // constraint-verification path unpacks the pols without loading the tree.
    bool constTreeResident = false;

    bool recursive;

    // This stream's aux-trace region intersects the streaming-commit slot area
    // (only ever set on first-GPU streams -- slots exist only there)
    bool overlapsStreamCommitRegion = false;

    // Field elements in this stream's slice of the unified buffer: a proof fits only when mapTotalN does.
    uint64_t auxTraceCapacity = 0;

#ifdef USE_CUDA_GRAPH
    std::unique_ptr<CudaGraphCache> graph_cache;
#endif

    std::mutex mutex_stream_selection;

    void initialize(uint64_t max_size_proof, uint32_t gpuId_, uint32_t localStreamId_, bool recursive_, uint64_t merkleTreeArity){
        uint64_t maxExps = PINNED_EXPS_SLOTS;
        cudaSetDevice(gpuId_);
        CHECKCUDAERR(cudaStreamCreate(&stream));
        timer.init(stream);
        gpuId = gpuId_;
        localStreamId = localStreamId_;
        recursive = recursive_;
        cudaEventCreate(&end_event);
        cudaEventCreate(&trace_copy_event);
        instanceId = -1;
        status = 0;
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_buffer_proof, max_size_proof * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_buffer_exps_params, maxExps * 2 * sizeof(DestParamsGPU)));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_buffer_exps_args, maxExps * sizeof(ExpsArguments)));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_params, sizeof(StepsParams)));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_aux_values, PINNED_AUX_VALUES_MAX * sizeof(Goldilocks::Element)));

        root = nullptr;
        pSetupCtx = nullptr;
        recurserId = "";
        witnessResident = false;
        proofBuffer = nullptr;
        airgroupId = UINT64_MAX;
        airId = UINT64_MAX;
        arity = merkleTreeArity;

    #ifdef USE_CUDA_GRAPH
        graph_cache = std::make_unique<CudaGraphCache>();
    #endif

        transcript = new TranscriptGL_GPU(merkleTreeArity,
                                    true,
                                    stream);

        transcript_helper = new TranscriptGL_GPU(merkleTreeArity,
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
        // end_event / trace_copy_event are created once in initialize() and destroyed in free();
        // cudaEventRecord overwrites them on each use, so there is no need to
        // destroy/recreate them on every per-instance reset.
        status = reset_status ? 0 : 3;

        root = nullptr;
        pSetupCtx = nullptr;
        proofBuffer = nullptr;

        // Clear stale open timer categories: a cancel mid-category leaves one open, and the next
        // job's stopCategory then mismatches and CHECKCUDAERR-aborts. Host-side only, no CUDA calls.
        timer.resetCategories();
    }

    // Invalidate the const-reuse identity so the next proof reloads constants.
    void invalidateContext(){
        airgroupId = UINT64_MAX;
        airId = UINT64_MAX;
        proofType = "";
        recurserId = "";
        witnessResident = false;
        constPolsOffset = UINT64_MAX;
        constRecurserId = "";
        constTreeResident = false;
    }

    // Claim this slot; true if it was already claimed, i.e. the unpacked const pols still
    // apply -- even to a different air sharing them. A change also drops constTreeResident.
    bool adoptFixedSlot(uint64_t offset, uint64_t auxOffset, bool aggBuffer, const string &recurser){
        if (constPolsOffset == offset && constAuxOffset == auxOffset && constAggBuffer == aggBuffer
            && constRecurserId == recurser) return true;
        constPolsOffset = offset;
        constAuxOffset = auxOffset;
        constAggBuffer = aggBuffer;
        constRecurserId = recurser;
        constTreeResident = false;
        return false;
    }

    // Nothing valid is cached any more: for paths that overwrite the aux trace without
    // repopulating the const pols.
    void dropFixedSlot(){
        constPolsOffset = UINT64_MAX;
        constRecurserId = "";
        constTreeResident = false;
    }

    void free(){
        cudaSetDevice(gpuId);
#ifdef USE_CUDA_GRAPH
        graph_cache.reset();
#endif
        cudaStreamDestroy(stream);
        cudaEventDestroy(end_event);
        cudaEventDestroy(trace_copy_event);
        cudaFreeHost(pinned_buffer_proof);
        cudaFreeHost(pinned_buffer_exps_params);
        cudaFreeHost(pinned_buffer_exps_args);
        cudaFreeHost(pinned_params);
        cudaFreeHost(pinned_aux_values);
    }
};

struct DeviceRecursiveFBuffers
{
    uint32_t gpuId;
    cudaStream_t stream;
    cudaStream_t stream_const_tree;
    TimerGPU timer;
    gl64_t *d_aux_trace;
    gl64_t *d_const_tree;
    RawFr::Element *d_verkey;  // Verification key on GPU
    uint8_t* pinnedBuffer;
    uint8_t* pinnedBufferConstTree;
    size_t pinnedBufferSize = 256 * 1024 * 1024;
    size_t aux_trace_size = 0;  // bytes of d_aux_trace when owned (standalone mode)
    bool owns_aux_trace;
    bool owns_const_tree;
    std::atomic<bool> const_tree_loaded{false};  // CPU flag: true when const tree copy is complete
    // Reusable allocations for proof generation
    StepsParams *params_pinned;
    Goldilocks::Element *pinned_exps_params;
    Goldilocks::Element *pinned_exps_args;
    StepsParams *d_params;
    ExpsArguments *d_expsArgs;
    DestParamsGPU *d_destParams;


    DeviceRecursiveFBuffers() : owns_aux_trace(true), owns_const_tree(true), d_verkey(nullptr), const_tree_loaded(false) {
        uint64_t maxExps = PINNED_EXPS_SLOTS;
        cudaStreamCreate(&stream);
        cudaStreamCreate(&stream_const_tree);
        timer.init(stream);
        
        CHECKCUDAERR(cudaMallocHost((void**)&pinnedBuffer, pinnedBufferSize));
        CHECKCUDAERR(cudaMallocHost((void**)&pinnedBufferConstTree, pinnedBufferSize));
        // Allocate reusable buffers
        CHECKCUDAERR(cudaMallocHost((void **)&params_pinned, sizeof(StepsParams)));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_exps_params, maxExps * 2 * sizeof(DestParamsGPU)));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_exps_args, maxExps * sizeof(ExpsArguments)));
        CHECKCUDAERR(cudaMalloc((void **)&d_params, sizeof(StepsParams)));
        CHECKCUDAERR(cudaMalloc((void **)&d_expsArgs, maxExps * sizeof(ExpsArguments)));
        CHECKCUDAERR(cudaMalloc((void **)&d_destParams, maxExps * 2 * sizeof(DestParamsGPU)));
    }
    
    ~DeviceRecursiveFBuffers() {
        cudaStreamDestroy(stream);
        cudaStreamDestroy(stream_const_tree);
        cudaFreeHost(pinnedBuffer);
        cudaFreeHost(pinnedBufferConstTree);
        cudaFreeHost(params_pinned);
        cudaFreeHost(pinned_exps_params);
        cudaFreeHost(pinned_exps_args);
        cudaFree(d_params);
        cudaFree(d_expsArgs);
        cudaFree(d_destParams);
        if (d_verkey) cudaFree(d_verkey);
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
    gl64_t **gpuMemoryBuffer;
    bool recursive;
    uint64_t max_size_proof;

    uint64_t constPolsSize;
    uint64_t unifiedBufferSize = 0;
    // Borrow flag for the FIRST GPU's unified buffer only (my_gpu_ids[0]).
    // 0 = free (proofman owns it), 1 = borrowed
    std::atomic<uint32_t> firstGpuBufferBorrowed{0};
    uint64_t pinned_size = 128 * 1024 * 1024; //256MB

    // Device-idle barrier. Worker: increment `device_active` THEN read `cancelled`. Teardown: raise
    // `cancelled` THEN wait for `device_active == 0`. seq_cst on both gives a total order so neither
    // side misses the other, letting teardown fence in-flight work before free. Covers only InFlightScope entries.
    std::atomic<int64_t> device_active{0};
    std::atomic<bool> cancelled{false};

    uint32_t  n_gpus;
    uint32_t* my_gpu_ids;
    uint32_t* gpus_g2l; 
    uint32_t n_total_streams;
    uint32_t n_streams;
    uint32_t n_recursive_streams;
    // Aux trace elements per non-recursive stream, largest class first; length n_streams. Owned here.
    uint64_t *aux_trace_sizes = nullptr;
    std::mutex *mutex_pinned;
    StreamData *streamsData;

    
    std::mutex stream_selection_mutex;

    bool packedTrace = false;

    // Streaming-commit slots (STREAM_COMMIT_SLOTS env, 0 = disabled), FIRST
    // GPU only -- the only one gpu-mops can borrow. Carved from the top of the unified buffer,
    // immediately below the const-pols aggregation region. They overlap the
    // Slot i starts at byte offset streamCommitFloorBytes + i * streamCommitSlotBytes
    // from gpuMemoryBuffer[0]. streamCommitFloorBytes is the ceiling gpu-mops
    // usage must stay under (UINT64_MAX when disabled, so comparisons degrade
    // to the const-pols one).
    uint64_t streamCommitSlots = 0;
    uint64_t streamCommitSlotBytes = 0;
    uint64_t streamCommitFloorBytes = UINT64_MAX;
    // Stashed at allocation for configure_stream_commit_slots' overlap computation. Non-recursive
    // streams differ in size, so only their total is meaningful; per-stream offsets are prefix sums.
    uint64_t auxTraceTotalBytes = 0;
    uint64_t auxTraceRecursiveBytes = 0;
    cudaStream_t *streamCommitStreams = nullptr;  // [streamCommitSlots], first GPU
    // Shared-hold of the overlapped legacy streams: the first in-flight slot
    // commit claims every overlapped stream's selection mutex, the last
    // releases them (see acquire/release in commit_witness_streaming_gpu).
    std::mutex streamCommitRegionMutex;
    uint32_t streamCommitInFlight = 0;
    // Quiesce: set by the gpu-mops borrower right before its FINAL planning
    // phase (whose host-paced micro-ops stretch ~40x under concurrent commit
    // load — see stream_commit_pause). While set, commit_witness_streaming
    // rejects new commits (-14, silent legacy fallback); cleared on the next
    // borrow acquire.
    std::atomic<uint32_t> streamCommitQuiesced{0};

    std::map<std::pair<uint64_t, uint64_t>, std::map<std::string, std::vector<AirInstanceInfo *>>> air_instances;
};

// RAII guard for the device-idle barrier. Construct at the top of a device entry (before touching
// the device) so teardown waits for this thread before freeing. Coverage is partial (not every
// entry constructs one). `cancelled()` lets an entry additionally bail out early when teardown has
// begun; increment-then-read pairs with teardown's raise-then-wait (seq_cst) so neither side misses
// the other. NOTE: no entry checks it yet — teardown currently relies on the refcount wait alone.
struct InFlightScope {
    DeviceCommitBuffers *d;
    explicit InFlightScope(DeviceCommitBuffers *d_) : d(d_) {
        d->device_active.fetch_add(1, std::memory_order_seq_cst);
    }
    ~InFlightScope() {
        d->device_active.fetch_sub(1, std::memory_order_seq_cst);
    }
    bool cancelled() const {
        return d->cancelled.load(std::memory_order_seq_cst);
    }
    InFlightScope(const InFlightScope &) = delete;
    InFlightScope &operator=(const InFlightScope &) = delete;
};

void copy_to_device_in_chunks(
    DeviceCommitBuffers* d_buffers,
    const void* src,
    void* dst,
    uint64_t total_size,
    uint64_t streamId,
    TimerGPU &timer);

void copy_to_device_in_chunks(
    const uint8_t* src,
    uint8_t* dst,
    uint64_t total_size_bytes,
    uint8_t* pinnedBuffer,
    uint64_t pinnedBufferSize,
    cudaStream_t stream);


void load_and_copy_to_device_in_chunks(
    DeviceCommitBuffers* d_buffers,
    const char* bufferPath,
    void* dst,
    uint64_t total_size,
    uint64_t streamId,
    uint64_t header_skip_bytes = 0
    );

#endif

// --- Data layout utilities
// Transpose row-major input -> the committed-section storage layout `layout` (the destination layout
// passed to getBufferOffset). Callers pass resolveLayout(nBits, nCols): ColMajor (flat) for most AIRs,
// ColMajorTiled for small high-column ones. Readers (expressions / Merkle) use the SAME layout.
__global__ void fromRowMajorToColMajor(
    const uint64_t nRows,
    const uint64_t nCols,
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    Layout layout
);

void fromRowMajorToColMajor(
    uint64_t nRows,
    uint64_t nCols,
    gl64_t* src,
    gl64_t* dst,
    Layout layout,
    cudaStream_t stream
);

#endif