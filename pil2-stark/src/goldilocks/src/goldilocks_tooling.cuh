#ifndef __GL64_GPU_CUH__
#define __GL64_GPU_CUH__

#include <cstdint>
#include <cassert>
#include <atomic>
#include <set>
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
// Bump-allocate from DeviceCommitBuffers::scratchArena (defined below); nullptr when full/absent.
void *deviceScratchArenaAlloc(void *d_buffers, uint64_t bytes);

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

    // No-const-buffer mode (PROOFMAN_NO_CONST_BUF=1): the packed const-pols
    // file identity, stashed at load time so gen_proof/commit_witness can stage
    // the blob into the prefetch zone on a slot switch instead of reading a
    // GPU-resident copy. const_pols_offset stays as the slot/reuse KEY only.
    std::string constPolsFile;
    uint64_t constPolsPackedBytes = 0;

    // Packed custom commits, in the same const-pols buffer. Reserved at load time (worst case,
    // words_per_row == nCols); customPolsPackedWords stays 0 until the blob is uploaded.
    uint64_t custom_pols_offset = 0;
    uint64_t customPolsReservedWords = 0;
    uint64_t customPolsPackedWords = 0;

    // Per-air pinned exps staging (deep pipeline only): CUDA-graph capture bakes
    // the buffer address into the graph's H2D nodes, and the content (per-stream
    // device pointers + per-air descriptors) must stay frozen for every later
    // replay. Sharing the per-stream buffer would make an air's capture overwrite
    // what an in-flight replay of the previous air still reads -- per-air copies
    // remove the hazard (and the stream-drain that guarded it). Allocated lazily
    // at the air's first pipelined launch; single-stream mode only (the content
    // embeds ONE stream's device pointers).
    Goldilocks::Element *pinnedExpsParams = nullptr;
    Goldilocks::Element *pinnedExpsArgs = nullptr;

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

    // Row bands the hash gates leave interior-blank, expanded on device after the trace copy
    // (see expandGateBandsGPU). A property of the circuit, so uploaded once with the setup.
    uint64_t *d_gate_bands = nullptr;   // n_gate_bands * 3 words: {row, kind, payload}
    uint64_t  n_gate_bands = 0;
    // The band section's aux word: setup parameters no kernel can recover from the trace (BLAKE3
    // packs LANES and the band width here; Poseidon writes 0).
    uint64_t  gate_band_aux = 0;
    // gate_bands::Family, kept untyped so this header does not depend on starkpil.
    uint64_t  gate_band_family = 0;
    // Dense BLAKE3 lookup counters, 2^17 + 2^16 words. The AIR holds these as two trace COLUMNS, so
    // counting straight into the trace makes every atomicAdd take a cache line of its own, shared with
    // witness cells other threads are writing. Accumulating here keeps the whole working set
    // contiguous and L2-resident and free of that false sharing; a trivial kernel scatters it into the
    // columns afterwards. The buffer itself lives on StreamData: this struct is per (air,
    // proofType, GPU), so one hung off it would be shared by every stream proving that air.

    /// Row stride of the HOST trace buffer, i.e. the exec map's width.
    ///
    /// `getCommitedPols` can only place what the circom witness carries, so it fills these columns and
    /// leaves the rest zero for the expander to rebuild -- which means copying the full width ships
    /// zeros the expander overwrites two kernels later. When this is narrower than the air's cm1 the
    /// host hands over a COMPACT buffer and the copy widens it on arrival. 0 means "not known", and
    /// the full-width path is used.
    uint64_t witness_map_cols = 0;

    /// Landing buffer for the compact host trace, `N * witness_map_cols` words. The copy has to be one
    /// contiguous run to get PCIe bandwidth: a 2D copy straight into the strided columns is slower than
    /// shipping the full width, because the rows are only `mapCols * 8` bytes -- too small for the DMA
    /// engine. So the transfer lands here and a kernel widens it on device, where the strided writes
    /// are cheap.
    uint64_t *d_witness_compact = nullptr;

    /// Allocate the landing buffer. Idempotent; must not run while work using it is in flight.
    /// `arena` (optional): bump-allocate from the unified buffer's scratch arena instead of
    /// cudaMalloc -- per-proof transient data, safe under the mops borrow.
    void set_witness_map(uint64_t mapCols, uint64_t nRows, uint64_t nCols, void *arena_owner = nullptr) {
        witness_map_cols = mapCols;
        if (d_witness_compact != nullptr) {
            if (!witness_compact_in_arena) CHECKCUDAERR(cudaFree(d_witness_compact));
            d_witness_compact = nullptr;
            witness_compact_in_arena = false;
        }
        if (mapCols > 0 && mapCols < nCols) {
            uint64_t bytes = nRows * mapCols * sizeof(uint64_t);
            if (arena_owner != nullptr) {
                d_witness_compact = (uint64_t *)deviceScratchArenaAlloc(arena_owner, bytes);
                witness_compact_in_arena = d_witness_compact != nullptr;
            }
            if (d_witness_compact == nullptr) {
                CHECKCUDAERR(cudaMalloc(&d_witness_compact, bytes));
            }
        }
    }
    bool witness_compact_in_arena = false;

    // Caller must have selected the target GPU. Replaces whatever was there.
    void set_gate_bands(const uint64_t *bands, uint64_t nBands, uint64_t aux, uint64_t family) {
        if (d_gate_bands != nullptr) {
            CHECKCUDAERR(cudaFree(d_gate_bands));
            d_gate_bands = nullptr;
        }
        n_gate_bands = nBands;
        gate_band_aux = aux;
        gate_band_family = family;
        if (nBands > 0) {
            CHECKCUDAERR(cudaMalloc(&d_gate_bands, nBands * 3 * sizeof(uint64_t)));
            CHECKCUDAERR(cudaMemcpy(d_gate_bands, bands, nBands * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
    }

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
        uint64_t num_batches = (setupCtx->starkInfo.openingPoints.size() + EVALS_OPENING_BATCH - 1) / EVALS_OPENING_BATCH;

        evalsInfo = new EvalInfo*[num_batches];
        evalsInfoSizes = new uint64_t[num_batches];
        numBatchesEvals = num_batches;

        uint64_t count = 0;
        for(uint64_t i = 0; i < setupCtx->starkInfo.openingPoints.size(); i += EVALS_OPENING_BATCH) {
            std::vector<int64_t> openingPoints;
            for(uint64_t j = 0; j < EVALS_OPENING_BATCH; ++j) {
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

        if (d_gate_bands != nullptr) {
            CHECKCUDAERR(cudaFree(d_gate_bands));
        }

        if (d_witness_compact != nullptr && !witness_compact_in_arena) {
            CHECKCUDAERR(cudaFree(d_witness_compact));
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
    // Stage-1 overlap lane (PROOFMAN_STAGE1_OVERLAP): the cm1 LDE+Merkle run here while the
    // main stream computes stage 2. cm1Fork orders the side work after the witness staging;
    // cm1LdeDone gates commit-2 (cm2-extended aliases the base trace the LDE reads, and Q
    // reads the LDE output); cm1TreeDone gates the query phase (first reader of mt1).
    cudaStream_t sideStream;
    // Phase-split lane: DEFAULT priority. The max-priority sideStream is for work the main
    // stream BLOCKS on (stage-1 float); phase-A is background fill with a huge window, and at
    // max priority it would preempt the predecessor's critical-path tail (measured: the split's
    // gain fully displaced). Equal priority = the same fair co-scheduling the 3-stream
    // reference gets.
    cudaStream_t phaseStream;
    
    // Custom-commit rebuild lane: LOWEST priority. The rebuild fills the proof's front-end gap
    // (witness H2D), so it must never preempt the proof's own kernels. customFixedFork orders it
    // after the previous proof's reads of custom_fixed; customFixedDone gates this proof's first use.
    cudaStream_t customStream;
    cudaEvent_t customFixedFork;
    cudaEvent_t customFixedDone;

    cudaEvent_t cm1Fork;
    cudaEvent_t cm1LdeDone;
    cudaEvent_t cm1TreeDone;
    // Query-phase split: FRI-tree openings run on the side stream while the main stream opens
    // the commitment trees; the buffers are disjoint per-tree slices of d_queries_buff.
    cudaEvent_t friQFork;
    cudaEvent_t friQDone;
    // Base/ext split phase choreography: phaseADone = this proof's early phase (base zone,
    // side stream) complete -- its phase-B waits it; baseFree = this proof's extends have
    // consumed the base-zone coefficients -- the NEXT same-air proof's phase-A waits it.
    // smallsUp = this proof's smalls upload (main stream) done -- its phase-A waits it.
    cudaEvent_t phaseADone;
    cudaEvent_t baseFree;
    cudaEvent_t smallsUp;
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
    // Dense scratch for whichever gate-band family needs one, gateBandScratchWordsGPU() words. Per
    // STREAM: the expander's memset/fill/scatter are ordered only within one. Allocated on this
    // stream's first commit of an air whose family asks for it.
    uint64_t *d_gate_band_scratch = nullptr;

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
    // Base/ext split parity transcripts: proof k+1's early phase seeds and pulls stage-2
    // challenges from ITS transcript while proof k's tail still folds on its own.
    TranscriptGL_GPU *transcript_parity;
    TranscriptGL_GPU *transcript_helper_parity;

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
    // moves it.
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

    // ---- Deep pipeline (single-stream zone mode, DeviceCommitBuffers::pipelineMode):
    // up to 2 basic proofs in flight on this stream. Completion metadata lives in this
    // 2-slot ring, harvested off the reserve path (reserve no longer host-syncs), and
    // enqueue-time host-written pinned staging (params / aux_values / proof) is
    // parity-sliced by launchSeq so proof N+1's CPU writes never race proof N's
    // still-pending async copies.
    struct PipelineSlot {
        int64_t instanceId = -1;
        uint64_t airgroupId = 0, airId = 0;
        std::string proofType = "basic";   // completion callback tag (ring carries recursives too)
        void *pSetupCtx = nullptr;
        uint64_t *proofBuffer = nullptr;
        std::string proofFile;
        Goldilocks::Element *pinnedProof = nullptr;
        cudaEvent_t done = nullptr;
    };
    PipelineSlot pipeSlots[2];
    uint32_t pipeHead = 0;
    uint32_t pipeCount = 0;      // guarded by pipeMutex
    uint64_t launchSeq = 0;      // single-writer (the launching worker)
    std::mutex pipeMutex;
    // Serializes harvesters: writeProof runs outside pipeMutex (it is slow and the
    // enqueue push must not block behind it), so without this two concurrent
    // harvesters (worker + settle poller) would both claim the same head slot,
    // double-pop, and underflow pipeCount into a livelock.
    std::mutex harvestMutex;
    uint64_t maxProofSize = 0;

    std::mutex mutex_stream_selection;

    void initialize(uint64_t max_size_proof, uint32_t gpuId_, uint32_t localStreamId_, bool recursive_, uint64_t merkleTreeArity){
        uint64_t maxExps = PINNED_EXPS_SLOTS;
        cudaSetDevice(gpuId_);
        CHECKCUDAERR(cudaStreamCreate(&stream));
        // Highest priority: the main stream BLOCKS on this stream's LDE (fence before the
        // quotient expressions), so its kernels must win co-scheduling over main's throughput
        // work or the fence stall eats the whole overlap.
        int prioLo = 0, prioHi = 0;
        CHECKCUDAERR(cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi));
        CHECKCUDAERR(cudaStreamCreateWithPriority(&sideStream, cudaStreamNonBlocking, prioHi));
        CHECKCUDAERR(cudaStreamCreateWithFlags(&phaseStream, cudaStreamNonBlocking));
        CHECKCUDAERR(cudaStreamCreateWithPriority(&customStream, cudaStreamNonBlocking, prioLo));
        CHECKCUDAERR(cudaEventCreateWithFlags(&customFixedFork, cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&customFixedDone, cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&cm1Fork, cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&cm1LdeDone, cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&cm1TreeDone, cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&friQFork, cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&friQDone, cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&phaseADone, cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&baseFree, cudaEventDisableTiming));
        CHECKCUDAERR(cudaEventCreateWithFlags(&smallsUp, cudaEventDisableTiming));
        timer.init(stream);
        gpuId = gpuId_;
        localStreamId = localStreamId_;
        recursive = recursive_;
        cudaEventCreate(&end_event);
        cudaEventCreate(&trace_copy_event);
        instanceId = -1;
        status = 0;
        // x2: parity slots for the deep pipeline (slot 0 is the only one used
        // outside pipeline mode, so single-proof behavior is unchanged).
        maxProofSize = max_size_proof;
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_buffer_proof, 2 * max_size_proof * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_buffer_exps_params, maxExps * 2 * sizeof(DestParamsGPU)));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_buffer_exps_args, maxExps * sizeof(ExpsArguments)));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_params, 2 * sizeof(StepsParams)));
        CHECKCUDAERR(cudaMallocHost((void **)&pinned_aux_values, 2 * PINNED_AUX_VALUES_MAX * sizeof(Goldilocks::Element)));
        for (int k = 0; k < 2; k++) {
            CHECKCUDAERR(cudaEventCreateWithFlags(&pipeSlots[k].done, cudaEventDisableTiming));
        }

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
        transcript_parity = new TranscriptGL_GPU(merkleTreeArity, true, stream);
        transcript_helper_parity = new TranscriptGL_GPU(merkleTreeArity, true, stream);

        // x2: parity slots -- under the base/ext phase split, the NEXT proof's early phase
        // H2Ds its argument structs while THIS proof's tail kernels still read theirs.
        CHECKCUDAERR(cudaMalloc(&params, 2 * sizeof(StepsParams)));
        CHECKCUDAERR(cudaMalloc(&d_destParams, 2 * 2 * sizeof(DestParamsGPU)));
        CHECKCUDAERR(cudaMalloc(&d_expsArgs, 2 * sizeof(ExpsArguments)));
    }

    ~StreamData() {
        delete transcript;
        delete transcript_helper;
        delete transcript_parity;
        delete transcript_helper_parity;
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
        cudaStreamDestroy(sideStream);
        cudaStreamDestroy(phaseStream);
        cudaStreamDestroy(customStream);
        cudaEventDestroy(customFixedFork);
        cudaEventDestroy(customFixedDone);
        cudaEventDestroy(cm1Fork);
        cudaEventDestroy(cm1LdeDone);
        cudaEventDestroy(cm1TreeDone);
        cudaEventDestroy(friQFork);
        cudaEventDestroy(friQDone);
        cudaEventDestroy(phaseADone);
        cudaEventDestroy(baseFree);
        cudaEventDestroy(smallsUp);
        cudaEventDestroy(end_event);
        cudaEventDestroy(trace_copy_event);
        if (d_gate_band_scratch != nullptr) {
            cudaFree(d_gate_band_scratch);
            d_gate_band_scratch = nullptr;
        }
        cudaFreeHost(pinned_buffer_proof);
        cudaFreeHost(pinned_buffer_exps_params);
        cudaFreeHost(pinned_buffer_exps_args);
        cudaFreeHost(pinned_params);
        cudaFreeHost(pinned_aux_values);
        for (int k = 0; k < 2; k++) {
            if (pipeSlots[k].done != nullptr) cudaEventDestroy(pipeSlots[k].done);
        }
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
    // Retirement events for the two pinned staging halves above (per GPU, index
    // 0 = pinned_buffer, 1 = pinned_buffer_extra). The chunked upload loops wait
    // on THESE before refilling a half, instead of cudaStreamSynchronize, which
    // under pipelining drains every queued kernel of the previous proof.
    cudaEvent_t (*pinned_copy_done)[2];
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
    // Mops-floor pad: dead space between the recursive aux area and the const regions so
    // the first-GPU borrow window (mem-ops planner) reaches PROOFMAN_MOPS_FLOOR_GB even
    // when the stream layout is small (blake, Main 2^23). Shifts the const offsets and
    // the stream-commit floor up; costs idle VRAM only.
    uint64_t mopsFloorPadBytes = 0;
    // Prefetch region: the zone (witness slots + fixed + packed) and the recursive-witness
    // slots live INSIDE the unified buffer, below the const regions — one planned budget,
    // and the space doubles as mops-borrow donor (keys invalidated on borrow release).
    gl64_t *prefetchRegionBase = nullptr;   // first GPU only
    uint64_t prefetchRegionBytes = 0;
    // Scratch arena over the mops-floor pad: bump allocations for per-air staging
    // buffers (compact witness). Pad space is dead otherwise; arena contents are
    // per-proof transient, so the mops borrow clobbering them is harmless.
    uint8_t *scratchArenaBase = nullptr;    // first GPU only
    uint64_t scratchArenaBytes = 0;
    uint64_t scratchArenaCursor = 0;
    std::mutex scratchArenaMutex;
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

    // Witness prefetch zone (PROOFMAN_PREFETCH, single-compute-stream mode):
    // the next basic instance's trace is uploaded on a dedicated copy stream
    // while the current proof computes; gen_proof drains it with one D2D and
    // records prefetchDrained so the next upload never overwrites live data.
    // FIRST GPU only. prefetchInstanceId == -1 means the zone is free.
    // Deep pipeline switch: set by the Rust proofs phase (single-stream zone mode
    // only), cleared at phase end. Read by the reserve paths and gen_proof.
    bool pipelineMode = false;
    gl64_t *prefetchZone = nullptr;
    uint64_t prefetchZoneBytes = 0;
    // Witness slots: 2 under the base/ext phase split (staging k+1 must not chain behind the
    // transpose of k), 1 otherwise. Per-slot ready/drained events and identity tags.
    uint32_t prefetchNSlots = 1;
    uint32_t prefetchStageSlot = 0;
    uint64_t prefetchSlotStride = 0; // elements between slot bases
    cudaStream_t prefetchStream = nullptr;
    cudaEvent_t prefetchReady[2] = {nullptr, nullptr};
    cudaEvent_t prefetchDrained[2] = {nullptr, nullptr};
    std::mutex prefetchMutex;
    int64_t prefetchInstanceId[2] = {-1, -1};
    uint64_t prefetchTraceBytes[2] = {0, 0};
    // Fixed half of the zone: next air's const TREE, staged from disk by the
    // prefetch worker thread. Key (-1,-1) = free. Its own events: tree and
    // witness prefetches interleave on the same copy stream.
    gl64_t *prefetchFixed = nullptr;
    // Packed const-pols segment of the zone (no-const-buffer mode): holds ONE
    // fixed-slot's packed blob, keyed by the slot offset; replaced on switch.
    gl64_t *prefetchPacked = nullptr;
    uint64_t prefetchPackedBytes = 0;
    int64_t packedSlotKey = -1;          // const_pols_offset of current content
    std::mutex packedMutex;
    cudaEvent_t packedReady = nullptr;   // staging complete (copy stream)
    cudaEvent_t packedDrained = nullptr; // last unpack read done (consumer stream)
    // Serializes users of the shared pinned staging pair (tree stager thread vs
    // inline packed/tree staging on the worker).
    std::mutex pinnedPairMutex;
    uint64_t prefetchFixedBytes = 0;
    // Recursive-witness staging (same copy stream): the NEXT recursive proof's
    // circom witness uploads while the current proof computes on the single
    // device stream. Two lazily-sized slots, keyed by the host trace pointer
    // (unique per in-flight witness buffer). recWitReady/recWitDrained pace
    // producer vs consumer exactly like the basic zone's events.
    gl64_t *recWitSlot[2] = {nullptr, nullptr};
    uint64_t recWitSlotBytes[2] = {0, 0};
    const void *recWitKey[2] = {nullptr, nullptr};
    uint64_t recWitBytes[2] = {0, 0};
    uint32_t recWitCursor = 0;
    cudaEvent_t recWitReady[2] = {nullptr, nullptr};
    cudaEvent_t recWitDrained[2] = {nullptr, nullptr};
    std::mutex recWitMutex;
    bool recWitCarved = false;   // slots carved from the prefetch region (fixed size)
    // Two 128 MB pinned staging chunks for file reads, used alternately so the
    // fread of chunk i+1 overlaps the H2D DMA of chunk i.
    Goldilocks::Element *prefetchPinned[2] = {nullptr, nullptr};
    cudaEvent_t prefetchFixedReady = nullptr;
    cudaEvent_t prefetchFixedDrained = nullptr;
    cudaEvent_t prefetchPinnedFree[2] = {nullptr, nullptr};  // pinned chunk reusable
    // Low-priority side stream for on-device rec-tree PRE-computation into the
    // fixed segment (prestage_rec_tree_compute): rebuild kernels fill SM gaps of
    // the in-flight proof instead of serializing ahead of the next one.
    cudaStream_t treeBuildStream = nullptr;
    TimerGPU treeBuildTimer;  // disabled: build kernels are not part of any proof window
    // Phase-B two-stream recursion (PROOFMAN_PHASE_B=1, single-basic-stream configs).
    // Two recursive streams whose aux buffers ALIAS the pre-const area of the unified
    // buffer ([0..A) and [A..2A), A = the rec1/rec2 class size): they cost no VRAM and
    // are only eligible while phaseBState==1 (after every basic+compressor completed,
    // when the basic stream's buffer and the prefetch region are dead). State machine:
    //   0 = phase A (basics/compressors on the basic stream; aliased rec streams OFF)
    //   1 = phase B (rec1/rec2 on the two aliased streams; non-recursive streams OFF)
    //   2 = final   (rec streams drained + OFF; basic stream back for VadcopFinal)
    std::atomic<uint32_t> phaseBState{0};
    bool phaseBAliased = false;              // aliased pair registered and safe to use
    uint8_t *phaseBSpareBase = nullptr;      // pre-const spare above the two aliases
    uint64_t phaseBSpareBytes = 0;
    uint8_t *phaseBMissScratch[2] = {nullptr, nullptr};  // compact-witness miss fallback per rec stream
    std::mutex prefetchFixedMutex;
    int64_t prefetchFixedAirgroup = -1;
    int64_t prefetchFixedAir = -1;
    uint64_t prefetchFixedSize = 0;      // 0 while staging is in flight
    // Set by gen_proof when it finds this air's staging still in flight (it
    // fell back to the legacy load): the stager drops the entry on completion
    // instead of leaving a never-consumed key that blocks the zone forever.
    bool prefetchFixedAbandon = false;

    std::map<std::pair<uint64_t, uint64_t>, std::map<std::string, std::vector<AirInstanceInfo *>>> air_instances;
};

inline void *deviceScratchArenaAlloc(void *d_buffers_, uint64_t bytes) {
    auto *db = (DeviceCommitBuffers *)d_buffers_;
    if (db == nullptr || db->scratchArenaBase == nullptr || bytes == 0) return nullptr;
    std::lock_guard<std::mutex> lk(db->scratchArenaMutex);
    uint64_t cur = (db->scratchArenaCursor + 255) & ~255ull;
    if (cur + bytes > db->scratchArenaBytes) return nullptr;
    db->scratchArenaCursor = cur + bytes;
    return db->scratchArenaBase + cur;
}

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
    TimerGPU &timer,
    bool categorize = true);

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