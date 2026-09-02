#ifndef GEN_PROOF_CUH
#define GEN_PROOF_CUH

#include "starks.hpp"
#include "starks_api_internal.cuh"
#include "cuda_utils.cuh"
#include "goldilocks_tooling.cuh"
#include "expressions_gpu.cuh"
#include "starks_gpu.cuh"
#include "hints.cuh"
#include "gpu_timer.cuh"
#include "cuda_graph_cache.cuh"
#include "proofman_sumcheck.cuh"
#include <iomanip>

// TOTO list: //rick
// carregar-me els d_trees
// _inplace not good name

// Fence-bisection knob: how far the previous proof must have progressed before the next
// same-air proof's phase-A may start (0 = after its extends [design], 4 = fully serial).
inline int splitAFenceLevel() {
    static const int lvl = [](){ const char* v = std::getenv("PROOFMAN_SPLIT_AFENCE"); return v ? atoi(v) : 0; }();
    return lvl;
}

// Stage-1 commit on a side stream, overlapped with stage 2+ (basics only). Off by default.
inline bool stage1OverlapEnabled() {
    static const bool enabled = []() {
        const char *v = std::getenv("PROOFMAN_STAGE1_OVERLAP");
        return v != nullptr && v[0] == '1';
    }();
    return enabled;
}

void calculateWitnessExpr_gpu(SetupCtx& setupCtx, StepsParams& h_params, StepsParams *d_params, ExpressionsGPU *expressionsCtxGPU, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream, uint64_t scratchShift = 0) {

    uint64_t nWitnessHints = setupCtx.expressionsBin.getNumberHintIdsByName("witness_calc");
    if(nWitnessHints > 0) {
        uint64_t witnessHints[nWitnessHints];
        setupCtx.expressionsBin.getHintIdsByName(witnessHints, "witness_calc");
        std::string hintFieldDest[nWitnessHints];
        std::string hintField[nWitnessHints];
        HintFieldOptions hintOptions[nWitnessHints];
        for(uint64_t i = 0; i < nWitnessHints; i++) {
            hintFieldDest[i] = "reference";
            hintField[i] = "expression";
            HintFieldOptions options;
            hintOptions[i] = options;
        }

        calculateExprGPU(setupCtx, h_params, d_params, nWitnessHints, witnessHints, hintFieldDest, hintField, hintOptions, expressionsCtxGPU, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, scratchShift);
    }
}

// The im_col/im_airval hints do not depend on `prod`: they read cm1 + stage-2 challenges and write
// the same "reference" destinations either way, so evaluating them inside calculateWitnessSTD_gpu ran
// the whole set twice whenever an AIR has both a gprod and a gsum column.
void calculateImHints_gpu(SetupCtx& setupCtx, StepsParams& h_params, StepsParams *d_params, ExpressionsGPU *expressionsCtxGPU, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream, uint64_t scratchShift = 0) {
    if(setupCtx.expressionsBin.getNumberHintIdsByName("gprod_col") == 0 && setupCtx.expressionsBin.getNumberHintIdsByName("gsum_col") == 0) return;

    uint64_t nImHints = setupCtx.expressionsBin.getNumberHintIdsByName("im_col");
    uint64_t nImHintsAirVals = setupCtx.expressionsBin.getNumberHintIdsByName("im_airval");
    uint64_t nImTotalHints = nImHints + nImHintsAirVals;
    if(nImTotalHints == 0) return;

    uint64_t imHints[nImTotalHints];
    setupCtx.expressionsBin.getHintIdsByName(imHints, "im_col");
    setupCtx.expressionsBin.getHintIdsByName(&imHints[nImHints], "im_airval");
    std::string hintFieldDest[nImTotalHints];
    std::string hintField1[nImTotalHints];
    std::string hintField2[nImTotalHints];
    HintFieldOptions hintOptions1[nImTotalHints];
    HintFieldOptions hintOptions2[nImTotalHints];
    for(uint64_t i = 0; i < nImTotalHints; i++) {
        hintFieldDest[i] = "reference";
        hintField1[i] = "numerator";
        hintField2[i] = "denominator";
        HintFieldOptions options1;
        HintFieldOptions options2;
        options2.inverse = true;
        hintOptions1[i] = options1;
        hintOptions2[i] = options2;
    }

    multiplyHintFieldsGPU(setupCtx, h_params, d_params, nImTotalHints, imHints, hintFieldDest, hintField1, hintField2, hintOptions1, hintOptions2, expressionsCtxGPU, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, scratchShift);
}

void calculateWitnessSTD_gpu(SetupCtx& setupCtx, StepsParams& h_params, StepsParams *d_params, bool prod, ExpressionsGPU *expressionsCtxGPU, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream, uint64_t scratchShift = 0, uint64_t hintValsOffset = 0) {

    std::string name = prod ? "gprod_col" : "gsum_col";
    if(setupCtx.expressionsBin.getNumberHintIdsByName(name) == 0) return;
    uint64_t hint[1];
    setupCtx.expressionsBin.getHintIdsByName(hint, name);

    HintFieldOptions options1;
    HintFieldOptions options2;
    options2.inverse = true;

    std::string hintFieldNameAirgroupVal = setupCtx.starkInfo.airgroupValuesMap.size() > 0 ? "result" : "";

    accMulHintFieldsGPU(setupCtx, h_params, d_params, hint[0], "reference", hintFieldNameAirgroupVal, "numerator_air", "denominator_air",options1, options2, !prod,expressionsCtxGPU, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, scratchShift, hintValsOffset);
    updateAirgroupValueGPU(setupCtx, h_params, d_params, hint[0], hintFieldNameAirgroupVal, "numerator_direct", "denominator_direct", options1, options2, !prod, expressionsCtxGPU, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, scratchShift, hintValsOffset);
}

void genProof_gpu(SetupCtx& setupCtx, gl64_t *d_aux_trace, gl64_t *d_const_pols, gl64_t *d_const_tree, char *constTreePath, uint32_t stream_id, uint64_t instance_id, DeviceCommitBuffers *d_buffers, AirInstanceInfo *air_instance_info, bool skipRecalculation, TimerGPU &timer, cudaStream_t stream, bool recursive = false, bool reuse_constants = false, bool splitPhases = false) {
    // Per-stream timer is reused: drop categories left open by an aborted job, and the load
    // phase's, so KERNELS CONTRIBUTIONS covers the proof window only.
    TimerResetCategoriesGPU(timer);
    TimerClearCategoriesGPU(timer);
    TimerStartGPU(timer, STARK_GPU_PROOF);
    TimerStartGPU(timer, STARK_STEP_0);

#ifdef USE_CUDA_GRAPH
    // Point the thread-local capture cache at THIS stream's cache for the duration of the
    // proof, and clear it on exit. Every capture region lives in this call tree, so clearing
    // here means any future beginCapture reached from another path (with current() unset)
    // faults loudly instead of silently reusing this stream's cache with another's buffers.
    // Phase-B aliased recursive streams run plain launches: concurrent first-use
    // captures on the two streams produced invalid proofs (root cause open); the
    // cross-stream overlap replaces what graph replay bought on one stream.
    // PROOFMAN_PAIR_GRAPHS=1 re-enables capture on the pair (testing the
    // capture-serialization fix); default stays plain launches there.
    static const bool pairGraphs = [] {
        const char *e = getenv("PROOFMAN_PAIR_GRAPHS");
        return e != nullptr && e[0] == '1';
    }();
    const bool phaseBRecStream = !pairGraphs &&
        d_buffers->phaseBAliased && d_buffers->streamsData[stream_id].recursive;
    cudagraph::current() = phaseBRecStream ? nullptr : d_buffers->streamsData[stream_id].graph_cache.get();
    cudaGetLastError();
    struct GraphCtxGuard {
        ~GraphCtxGuard() { cudagraph::current() = nullptr; }
    } graphCtxGuard;
#endif

    uint64_t countId = 0;
    // Per-air identity for every capture-region key on this proof (setups are stable
    // heap objects for the process lifetime; caches are per-stream).
    //
    // Capture-region key tags (ASCII), all mixed with graphCtxId so every
    // (air, stream, region) gets its own graph:
    //   0x57455850 "WEXP"  witness expressions
    //   0x434d5431 "CM1"   commit stage 1 (+ recursive, skipRecalculation in key)
    //   0x53544432 "STD2"  stage-2 getFields + gsum/gprod + im hints
    //   0x494d504c "IMPL"  im pols
    //   0x434d5432 "CM2"   commit stage 2 + airValues transcript puts
    //   0x51455850 "QEXP"  Q expression
    //   0x434d5451 "CMQ"   commit stage Q
    //   0x4556414c "EVAL"  evals memset + LEv/evmap + evals transcript + FRI challenges
    //   0x46524950 "FRIP"  calculateXis + computeX
    //   0x46524558 "FREX"  FRI expression
    //   0x465249   "FRI"   FRI fold+merkelize step (+ step in key)
    //   0x4752494e "GRIN"  grinding (+ powBits in key)
    //   0x515559   "QUY"   query proofs (+ d_const_tree in key: preloaded trees repoint it)
    //   0x57455843 "WEXC"  contributions witness expressions   (commit_witness_gpu, starks_api.cu)
    //   0x574c4445 "WLDE"  contributions LDE + Merkle + root   (commit_witness_gpu, starks_api.cu)

    // Deep pipeline: parity-sliced pinned staging (see StreamData::PipelineSlot).
    // launchSeq increments at the ring push AFTER this call, so the parity here
    // matches the slot the caller records for this proof. Recursives on the shared
    // stream ride the ring too; on recursive-class streams (no ring) launchSeq
    // stays 0, so this still resolves to slot 0 there.
    // skipRecalculation proofs are RESIDENT: their early state (stage-1, hint values)
    // was written by the contributions phase at UNSHIFTED offsets, so they must run at
    // parity 0 -- and they need no early-phase staging (uploads are stream-ordered).
    const uint32_t pipeSlot = (d_buffers->pipelineMode && !skipRecalculation)
        ? (uint32_t)(d_buffers->streamsData[stream_id].launchSeq & 1) : 0;
    // Base/ext split parity: odd-parity proofs relocate the early-phase smalls (publics..
    // challenges) and the expression scratch to their parity copies, so the next proof's early
    // phase never touches this proof's live state. Must match gen_proof_gpu's upload shift.
    const uint64_t smallsShift = (setupCtx.starkInfo.baseSplit && pipeSlot)
        ? setupCtx.starkInfo.mapOffsets[std::make_pair("smalls_parity", false)] - setupCtx.starkInfo.mapOffsets[std::make_pair("publics", false)]
        : 0;
    const uint64_t scratchShift = (setupCtx.starkInfo.baseSplit && pipeSlot) ? setupCtx.starkInfo.expsScratchShift() : 0;
    // Base-zone scratch for the gsum accumulation (stock code borrows the ("q",true) region,
    // which the previous proof's tail still folds under the phase split).
    const uint64_t hintValsOffset = setupCtx.starkInfo.baseSplit
        ? setupCtx.starkInfo.mapOffsets[std::make_pair(pipeSlot ? "hint_vals_parity" : "hint_vals", false)]
        : 0;
    // (declared after the parity shifts it keys on)
    const uint64_t graphCtxId = (uint64_t)(uintptr_t)&setupCtx ^ (smallsShift ? 0x9E3779B97F4A7C15ULL : 0ULL);
    StepsParams *params_pinned = d_buffers->streamsData[stream_id].pinned_params + pipeSlot;
    Goldilocks::Element *proof_buffer_pinned = d_buffers->streamsData[stream_id].pinned_buffer_proof
        + (uint64_t)pipeSlot * d_buffers->streamsData[stream_id].maxProofSize;
    // Deep pipeline: per-air pinned exps staging (frozen post-capture; see
    // AirInstanceInfo). Shared per-stream buffers otherwise.
    // Base/ext split parity: the odd-parity proof's per-proof device/pinned argument state
    // must be its own copy -- the even proof's tail kernels still read theirs while this
    // proof's early phase stages and launches.
    const bool parityOdd = setupCtx.starkInfo.baseSplit && (d_buffers->pipelineMode && !recursive)
        && (d_buffers->streamsData[stream_id].launchSeq & 1);
    const bool perAirExps = d_buffers->pipelineMode && !recursive && air_instance_info->pinnedExpsParams != nullptr;
    Goldilocks::Element *pinned_exps_params = perAirExps
        ? air_instance_info->pinnedExpsParams + (parityOdd ? (uint64_t)PINNED_EXPS_SLOTS * 2 * sizeof(DestParamsGPU) / sizeof(Goldilocks::Element) : 0)
        : d_buffers->streamsData[stream_id].pinned_buffer_exps_params;
    Goldilocks::Element *pinned_exps_args = perAirExps
        ? air_instance_info->pinnedExpsArgs + (parityOdd ? (uint64_t)PINNED_EXPS_SLOTS * sizeof(ExpsArguments) / sizeof(Goldilocks::Element) : 0)
        : d_buffers->streamsData[stream_id].pinned_buffer_exps_args;
    TranscriptGL_GPU *d_transcript = parityOdd ? d_buffers->streamsData[stream_id].transcript_parity
                                               : d_buffers->streamsData[stream_id].transcript;
    TranscriptGL_GPU *d_transcript_helper = parityOdd ? d_buffers->streamsData[stream_id].transcript_helper_parity
                                                      : d_buffers->streamsData[stream_id].transcript_helper;
    StepsParams *d_params =  d_buffers->streamsData[stream_id].params + (parityOdd ? 1 : 0);
    ExpsArguments *d_expsArgs = d_buffers->streamsData[stream_id].d_expsArgs + (parityOdd ? 1 : 0);
    DestParamsGPU *d_destParams = d_buffers->streamsData[stream_id].d_destParams + (parityOdd ? 2 : 0);

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    Goldilocks::Element *pConstPolsExtendedTreeAddress = (Goldilocks::Element *)d_const_tree;
    Goldilocks::Element *pCustomCommitsFixed = (Goldilocks::Element *)d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("custom_fixed", false)];
    
    Starks<Goldilocks::Element> starks(setupCtx, nullptr, nullptr, false, false);
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->setSource(pConstPolsExtendedTreeAddress);
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->setNodes(&pConstPolsExtendedTreeAddress[setupCtx.starkInfo.nConstants * NExtended]);
    for(uint64_t i = 0; i < setupCtx.starkInfo.customCommits.size(); i++) {
        uint64_t nCols = setupCtx.starkInfo.mapSectionsN[setupCtx.starkInfo.customCommits[i].name + "0"];
            starks.treesGL[setupCtx.starkInfo.nStages + 2 + i]->setSource(&pCustomCommitsFixed[N * nCols]);
            starks.treesGL[setupCtx.starkInfo.nStages + 2 + i]->setNodes(&pCustomCommitsFixed[(N + NExtended) * nCols]);
    }

    uint64_t nFieldElements = setupCtx.starkInfo.starkStruct.verificationHashType == std::string("BN128") ? 1 : HASH_SIZE;
    
    uint64_t offsetCm1 = setupCtx.starkInfo.mapOffsets[std::make_pair("cm1", false)];
    uint64_t offsetPublicInputs = setupCtx.starkInfo.mapOffsets[std::make_pair("publics", false)] + smallsShift;
    uint64_t offsetAirgroupValues = setupCtx.starkInfo.mapOffsets[std::make_pair("airgroupvalues", false)] + smallsShift;
    uint64_t offsetAirValues = setupCtx.starkInfo.mapOffsets[std::make_pair("airvalues", false)] + smallsShift;
    uint64_t offsetProofValues = setupCtx.starkInfo.mapOffsets[std::make_pair("proofvalues", false)] + smallsShift;
    uint64_t offsetEvals = setupCtx.starkInfo.mapOffsets[std::make_pair("evals", false)] + smallsShift;
    uint64_t offsetChallenges = setupCtx.starkInfo.mapOffsets[std::make_pair("challenges", false)] + smallsShift;
    uint64_t offsetXDivXSub = setupCtx.starkInfo.mapOffsets[std::make_pair("xdivxsub", false)];
    uint64_t offsetFriQueries = setupCtx.starkInfo.mapOffsets[std::make_pair("fri_queries", false)];
    uint64_t offsetChallenge = setupCtx.starkInfo.mapOffsets[std::make_pair("challenge", false)] + smallsShift;
    uint64_t offsetNonce = setupCtx.starkInfo.mapOffsets[std::make_pair("nonce", false)] + smallsShift;
    uint64_t offsetNonceBlocks = setupCtx.starkInfo.mapOffsets[std::make_pair("nonce_blocks", false)] + smallsShift;
    uint64_t offsetInputHashNonce = setupCtx.starkInfo.mapOffsets[std::make_pair("input_hash_nonce", false)] + smallsShift;
    uint64_t offsetProofQueries = setupCtx.starkInfo.mapOffsets[std::make_pair("proof_queries", false)];
    uint64_t offsetConstPols = setupCtx.starkInfo.mapOffsets[std::make_pair("const", false)];

    Goldilocks::Element *packed_const_pols = (Goldilocks::Element *)d_const_pols;
    Goldilocks::Element *d_const_pols_unpacked = (Goldilocks::Element *)d_aux_trace + offsetConstPols;
    if(!reuse_constants) {
        uint64_t* d_num_packed_words = (uint64_t*) d_const_pols;
        unpack_fixed(d_num_packed_words, (uint64_t*)(packed_const_pols + 1), (uint64_t*)(packed_const_pols + 1 + setupCtx.starkInfo.nConstants), (uint64_t*)d_const_pols_unpacked, setupCtx.starkInfo.nConstants, N, stream, timer);
        CHECKCUDAERR(cudaGetLastError());
        // No-const-buffer mode: mark the packed zone segment's last read so the
        // next slot's staging cannot overwrite it mid-unpack.
        if (d_buffers->prefetchPacked != nullptr && (gl64_t *)packed_const_pols == d_buffers->prefetchPacked) {
            CHECKCUDAERR(cudaEventRecord(d_buffers->packedDrained, stream));
        }
    }

    StepsParams h_params = {
        trace : (Goldilocks::Element *)d_aux_trace + offsetCm1,
        aux_trace : (Goldilocks::Element *)d_aux_trace,
        publicInputs : (Goldilocks::Element *)d_aux_trace + offsetPublicInputs,
        proofValues : (Goldilocks::Element *)d_aux_trace + offsetProofValues,
        challenges : (Goldilocks::Element *)d_aux_trace + offsetChallenges,
        airgroupValues : (Goldilocks::Element *)d_aux_trace + offsetAirgroupValues,
        airValues : (Goldilocks::Element *)d_aux_trace + offsetAirValues,
        evals : (Goldilocks::Element *)d_aux_trace + offsetEvals,
        xDivXSub : (Goldilocks::Element *)d_aux_trace + offsetXDivXSub,
        pConstPolsAddress: d_const_pols_unpacked,
        pConstPolsExtendedTreeAddress,
        pCustomCommitsFixed,
    };

    memcpy(params_pinned, &h_params, sizeof(StepsParams));
    
    // Split phases: the early phase's expression kernels read d_params on the SIDE stream --
    // upload there so it lands before them without waiting the previous proof's tail on main.
    // (bit 2 of PROOFMAN_SPLIT_EARLY_UPLOAD; bisection knob)
    static const int earlyUpG = [](){ const char* v = std::getenv("PROOFMAN_SPLIT_EARLY_UPLOAD"); return v ? atoi(v) : 0; }();
    CHECKCUDAERR(cudaMemcpyAsync(d_params, params_pinned, sizeof(StepsParams), cudaMemcpyHostToDevice,
                                 (splitPhases && (earlyUpG & 2)) ? d_buffers->streamsData[stream_id].phaseStream : stream));
    
    Goldilocks::Element *d_challenge = (Goldilocks::Element *)d_aux_trace + offsetChallenge;
    
    Goldilocks::Element *d_nonce = (Goldilocks::Element *)d_aux_trace + offsetNonce;

    Goldilocks::Element *d_nonceBlocks = (Goldilocks::Element *)d_aux_trace + offsetNonceBlocks;

    uint64_t *friQueries_gpu = (uint64_t *)d_aux_trace + offsetFriQueries;

    gl64_t *d_queries_buff = (gl64_t *)d_aux_trace + offsetProofQueries;
    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    uint64_t nTreesFRI = setupCtx.starkInfo.starkStruct.steps.size() - 1;

    TimerStartCategoryGPU(timer, TRANSCRIPT);
    if (!splitPhases) {
    d_transcript->reset(stream);
    if (recursive) {
        d_transcript->put(air_instance_info->verkeyRoot, HASH_SIZE, stream);
        if (setupCtx.starkInfo.nPublics > 0)
        {
            if (!setupCtx.starkInfo.starkStruct.hashCommits)
            {
                d_transcript->put(h_params.publicInputs, setupCtx.starkInfo.nPublics, stream);
            }
            else
            {
                calculateHash(d_transcript_helper, h_params.challenges, setupCtx, h_params.publicInputs, setupCtx.starkInfo.nPublics, stream);
                d_transcript->put(h_params.challenges, HASH_SIZE, stream);
            }
        }
    } else {
       d_transcript->put(d_challenge, FIELD_EXTENSION, stream);
    }
    }
    TimerStopCategoryGPU(timer, TRANSCRIPT);

    if (!skipRecalculation && !splitPhases) {
        uint64_t offsetCm1Extended = setupCtx.starkInfo.mapOffsets[std::make_pair("cm1", true)];
        if (d_buffers->packedTrace && air_instance_info->is_packed) {
            uint64_t nCols = setupCtx.starkInfo.mapSectionsN["cm1"];
            unpack_trace(air_instance_info, (uint64_t*)h_params.aux_trace + offsetCm1Extended, (uint64_t*)h_params.trace, nCols, N, stream, timer);
        } else {
            fromRowMajorToColMajor(N, setupCtx.starkInfo.mapSectionsN["cm1"], (gl64_t *)h_params.aux_trace + offsetCm1Extended, (gl64_t*)h_params.trace, resolveLayout(setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.mapSectionsN["cm1"]), stream);
        }
    }
    TimerStopGPU(timer, STARK_STEP_0);
    
    TimerStartGPU(timer, STARK_COMMIT_STAGE_1);
    StreamData &sdPh = d_buffers->streamsData[stream_id];
    StreamData &sdOv = sdPh;
    // Stage-1 side-stream overlap is subsumed by the phase split (its halves live in the
    // phases); active only in the non-split path.
    bool overlapStage1 = stage1OverlapEnabled() && !recursive && !skipRecalculation && !splitPhases;
    if (splitPhases) {
        // ================= PHASE A (base zone, side stream) =================
        // The zone->trace transpose was already enqueued on the side stream by gen_proof_gpu
        // (under the prefetch mutex). Everything here reads/writes ONLY base-zone regions and
        // the parity smalls/scratch, so it may execute while the PREVIOUS proof's tail still
        // owns the ext zone.
        cudaStream_t aStream = sdPh.phaseStream;
        // Smalls (publics/airvalues/globalChallenge) upload runs on the main stream.
        CHECKCUDAERR(cudaStreamWaitEvent(aStream, sdPh.smallsUp, 0));
        PROOFMAN_SUMCHECK("splitA_trace_i%u", (gl64_t*)h_params.trace, N * setupCtx.starkInfo.mapSectionsN["cm1"], aStream, (unsigned)instance_id);
        PROOFMAN_SUMCHECK("splitA_smalls_i%u", (gl64_t*)h_params.publicInputs, setupCtx.starkInfo.nPublics + setupCtx.starkInfo.proofValuesSize + setupCtx.starkInfo.airgroupValuesSize + setupCtx.starkInfo.airValuesSize + FIELD_EXTENSION, aStream, (unsigned)instance_id);
        d_transcript->reset(aStream);
        d_transcript->put(d_challenge, FIELD_EXTENSION, aStream);
        calculateWitnessExpr_gpu(setupCtx, h_params, d_params, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, aStream, scratchShift);
        for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
            if(setupCtx.starkInfo.challengesMap[i].stage == 2) {
                d_transcript->getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], aStream);
            }
        }
        calculateWitnessSTD_gpu(setupCtx, h_params, d_params, true, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, aStream, scratchShift, hintValsOffset);
        calculateWitnessSTD_gpu(setupCtx, h_params, d_params, false, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, aStream, scratchShift, hintValsOffset);
        calculateImPolsExpressions(setupCtx, air_instance_info->expressions_gpu, h_params, d_params, 2, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, aStream, scratchShift);
        PROOFMAN_SUMCHECK("splitA_wexpstd_i%u", (gl64_t*)h_params.trace, N * setupCtx.starkInfo.mapSectionsN["cm1"], aStream, (unsigned)instance_id);
        PROOFMAN_SUMCHECK("splitA_cm2b_i%u", (gl64_t*)h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("cm2", false)], N * setupCtx.starkInfo.mapSectionsN["cm2"], aStream, (unsigned)instance_id);
        {
            NTTGoldilocksGPU nttA;
            // In-place iNTTs AFTER the last base-domain readers (STD2/imPols): evaluations
            // become bit-reversed coefficients that phase B extends from.
            nttA.inttToCoeffsRevColMajor((gl64_t*)h_params.trace, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.mapSectionsN["cm1"], aStream);
            nttA.inttToCoeffsRevColMajor((gl64_t*)h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("cm2", false)], setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.mapSectionsN["cm2"], aStream);
        }
        PROOFMAN_SUMCHECK("splitA_intt1_i%u", (gl64_t*)h_params.trace, N * setupCtx.starkInfo.mapSectionsN["cm1"], aStream, (unsigned)instance_id);
        PROOFMAN_SUMCHECK("splitA_intt2_i%u", (gl64_t*)h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("cm2", false)], N * setupCtx.starkInfo.mapSectionsN["cm2"], aStream, (unsigned)instance_id);
        CHECKCUDAERR(cudaEventRecord(sdPh.phaseADone, aStream));
        // ================= PHASE B (ext zone, main stream) =================
        CHECKCUDAERR(cudaStreamWaitEvent(stream, sdPh.phaseADone, 0));
        {
            NTTGoldilocksGPU nttB;
            uint64_t NExtB = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
            Goldilocks::Element *dstGL = (Goldilocks::Element*)h_params.aux_trace;
            // stage 1: extend + tree (root1 is never absorbed for basics)
            {
                uint64_t nC = setupCtx.starkInfo.mapSectionsN["cm1"];
                uint64_t offD = setupCtx.starkInfo.mapOffsets[std::make_pair("cm1", true)];
                Goldilocks::Element *pNodes = dstGL + setupCtx.starkInfo.mapOffsets[std::make_pair("mt1", true)];
                starks.treesGL[0]->setSource(dstGL + offD);
                starks.treesGL[0]->setNodes(pNodes);
                TimerStartCategoryGPU(timer, NTT);
                nttB.extendFromCoeffsColMajor((gl64_t*)h_params.aux_trace + offD, (const gl64_t*)h_params.trace, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, nC, stream);
                TimerStopCategoryGPU(timer, NTT);
                TimerStartCategoryGPU(timer, MERKLE_TREE);
                buildMerkleTreeGPU(setupCtx.starkInfo.starkStruct.merkleTreeArity, (uint64_t*)pNodes, (uint64_t*)(dstGL + offD), nC, NExtB, resolveLayout(setupCtx.starkInfo.starkStruct.nBits, nC), stream);
                TimerStopCategoryGPU(timer, MERKLE_TREE);
            }
            // stage 2: extend, THEN release the base zone (both coefficient reads done), then tree + root2
            {
                uint64_t nC = setupCtx.starkInfo.mapSectionsN["cm2"];
                uint64_t offS = setupCtx.starkInfo.mapOffsets[std::make_pair("cm2", false)];
                uint64_t offD = setupCtx.starkInfo.mapOffsets[std::make_pair("cm2", true)];
                Goldilocks::Element *pNodes = dstGL + setupCtx.starkInfo.mapOffsets[std::make_pair("mt2", true)];
                starks.treesGL[1]->setSource(dstGL + offD);
                starks.treesGL[1]->setNodes(pNodes);
                TimerStartCategoryGPU(timer, NTT);
                nttB.extendFromCoeffsColMajor((gl64_t*)h_params.aux_trace + offD, (const gl64_t*)(h_params.aux_trace + offS), setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, nC, stream);
                TimerStopCategoryGPU(timer, NTT);
                if (splitAFenceLevel() == 0) CHECKCUDAERR(cudaEventRecord(sdPh.baseFree, stream));
                TimerStartCategoryGPU(timer, MERKLE_TREE);
                buildMerkleTreeGPU(setupCtx.starkInfo.starkStruct.merkleTreeArity, (uint64_t*)pNodes, (uint64_t*)(dstGL + offD), nC, NExtB, resolveLayout(setupCtx.starkInfo.starkStruct.nBits, nC), stream);
                TimerStopCategoryGPU(timer, MERKLE_TREE);
                uint64_t tree_size = starks.treesGL[1]->getNumNodes(NExtB);
                PROOFMAN_SUMCHECK("splitB_ext1_i%u", (gl64_t*)h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("cm1", true)], NExtB * setupCtx.starkInfo.mapSectionsN["cm1"], stream, (unsigned)instance_id);
                PROOFMAN_SUMCHECK("splitB_ext2_i%u", (gl64_t*)h_params.aux_trace + offD, NExtB * nC, stream, (unsigned)instance_id);
                d_transcript->put(&pNodes[tree_size - HASH_SIZE], HASH_SIZE, stream);
            }
            uint64_t a = 0;
            for(uint64_t i = 0; i < setupCtx.starkInfo.airValuesMap.size(); i++) {
                if(setupCtx.starkInfo.airValuesMap[i].stage == 1) a++;
                if(setupCtx.starkInfo.airValuesMap[i].stage == 2) {
                    d_transcript->put(&h_params.airValues[a], FIELD_EXTENSION, stream);
                    a += 3;
                }
            }
        }
    } else {
    cudagraph::run(cudagraph::key(0x57455850ULL ^ graphCtxId), countId, stream, [&] {
        calculateWitnessExpr_gpu(setupCtx, h_params, d_params, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, scratchShift);
    });
    // Stage-1 overlap (PROOFMAN_STAGE1_OVERLAP): basics never absorb root1 into the transcript,
    // so the stage-1 commit is off the Fiat-Shamir critical path (see hoisted flag above).
    if (overlapStage1) {
        CHECKCUDAERR(cudaEventRecord(sdOv.cm1Fork, stream));
        CHECKCUDAERR(cudaStreamWaitEvent(sdOv.sideStream, sdOv.cm1Fork, 0));
        // Not graph-wrapped: capture runs on the main stream and would not see side-stream work.
        extendAndMerkelize_inplace(1, setupCtx, starks.treesGL, (gl64_t*) h_params.trace, (gl64_t*)h_params.aux_trace, nullptr, false, timer, sdOv.sideStream, sdOv.cm1LdeDone);
        CHECKCUDAERR(cudaEventRecord(sdOv.cm1TreeDone, sdOv.sideStream));
    } else {
    cudagraph::run(cudagraph::key(0x434d5431ULL ^ graphCtxId, recursive, skipRecalculation), countId, stream, [&] {
    // The transcript differs between the two, but skipRecalculation is the caller's answer to
    // "is the witness already committed" and must be honoured either way.
    commitStage_inplace(1, setupCtx, starks.treesGL, (gl64_t*) h_params.trace, (gl64_t*)h_params.aux_trace, recursive ? d_transcript : nullptr, skipRecalculation, timer, stream);
    });
    }
    TimerStopGPU(timer, STARK_COMMIT_STAGE_1);

    TimerStartGPU(timer, STARK_CALCULATE_WITNESS_STD);
    cudagraph::run(cudagraph::key(0x53544432ULL ^ graphCtxId), countId, stream, [&] {
    TimerStartCategoryGPU(timer, TRANSCRIPT);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 2) {
            d_transcript->getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    TimerStopCategoryGPU(timer, TRANSCRIPT);
    calculateImHints_gpu(setupCtx, h_params, d_params, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, scratchShift);
    calculateWitnessSTD_gpu(setupCtx, h_params, d_params, true, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, scratchShift, hintValsOffset);
    calculateWitnessSTD_gpu(setupCtx, h_params, d_params, false, air_instance_info->expressions_gpu, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, scratchShift, hintValsOffset);

    });
    TimerStopGPU(timer, STARK_CALCULATE_WITNESS_STD);

    // Own capture region so the section timers stay outside every region body: a replay returns
    // before the body, so a start/stop pair split across it loses the stop event.
    TimerStartGPU(timer, CALCULATE_IM_POLS);
    // NOTE: scratchShift is 0 unless the single-proof pipeline alternates slots; if that mode
    // ever runs with cudagraphs on, the shift must join the graph key (baked otherwise).
    cudagraph::run(cudagraph::key(0x494d504cULL ^ graphCtxId), countId, stream, [&] {
        calculateImPolsExpressions(setupCtx, air_instance_info->expressions_gpu, h_params, d_params, 2, d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, scratchShift);
        PROOFMAN_SUMCHECK("plain_im_i%u", (gl64_t*)h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("cm2", false)], N * setupCtx.starkInfo.mapSectionsN["cm2"], stream, (unsigned)instance_id);
    });
    TimerStopGPU(timer, CALCULATE_IM_POLS);
    
    TimerStartGPU(timer, STARK_COMMIT_STAGE_2);
    // Stage-1 overlap fence #1: only needed here when cm2-extended aliases cm1(false) (the side
    // LDE's src) -- with the un-aliased layout commit-2 is conflict-free and the LDE fence moves
    // to the quotient expressions (the first reader of the stage-1 LDE output).
    if (overlapStage1 && !setupCtx.starkInfo.cm2Unaliased) {
        CHECKCUDAERR(cudaStreamWaitEvent(stream, sdOv.cm1LdeDone, 0));
    }
    cudagraph::run(cudagraph::key(0x434d5432ULL ^ graphCtxId), countId, stream, [&] {
    commitStage_inplace(2, setupCtx, starks.treesGL, (gl64_t*)h_params.trace, (gl64_t*)h_params.aux_trace, d_transcript, false, timer, stream);

    uint64_t a = 0;
    TimerStartCategoryGPU(timer, TRANSCRIPT);
    for(uint64_t i = 0; i < setupCtx.starkInfo.airValuesMap.size(); i++) {
        if(setupCtx.starkInfo.airValuesMap[i].stage == 1) a++;
        if(setupCtx.starkInfo.airValuesMap[i].stage == 2) {
            d_transcript->put(&h_params.airValues[a], FIELD_EXTENSION, stream);
            a += 3;
        }
    }
    TimerStopCategoryGPU(timer, TRANSCRIPT);
    });
    }
    TimerStopGPU(timer, STARK_COMMIT_STAGE_2);
    TimerStartGPU(timer, STARK_STEP_Q);
    TimerStartCategoryGPU(timer, TRANSCRIPT);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1) {
            d_transcript->getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    TimerStopCategoryGPU(timer, TRANSCRIPT);
    PROOFMAN_SUMCHECK("tail_qch_i%u", (gl64_t*)h_params.challenges, setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION, stream, (unsigned)instance_id);
    uint64_t zi_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("zi", true)];
    // The zerofier is what the Q expression divides by, and it spans the extended domain.
    TimerStartCategoryGPU(timer, EXPRESSIONS);
    computeZerofier(h_params.aux_trace + zi_offset, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, stream);
    TimerStopCategoryGPU(timer, EXPRESSIONS);

    if (setupCtx.starkInfo.calculateFixedExtended && !reuse_constants) {
        TimerStartGPU(timer, FIXED_POLS_TREE);
        extendAndMerkelizeFixed(setupCtx, h_params.pConstPolsAddress, pConstPolsExtendedTreeAddress,
                                !setupCtx.starkInfo.constPolsAliasTree, timer, stream);
        TimerStopGPU(timer, FIXED_POLS_TREE);
    }

    TimerStartGPU(timer, STARK_QUOTIENT_POLYNOMIAL);
    // Stage-1 overlap fence #1 (un-aliased layout): quotient expressions read cm1-extended.
    if (overlapStage1 && setupCtx.starkInfo.cm2Unaliased) {
        CHECKCUDAERR(cudaStreamWaitEvent(stream, sdOv.cm1LdeDone, 0));
    }
    // (legacy base-zone release moved to just before setProof: a residual boundary race --
    // intermittent Binary#1 'Invalid evaluations' -- survives the earlier release point, and
    // legacy predecessors are only the first proof of each same-air run, so the conservative
    // fence costs overlap at ~5 boundary pairs per block.)
    if (setupCtx.starkInfo.baseSplit && !recursive &&
        ((splitPhases && splitAFenceLevel() == 1) || (!splitPhases && splitAFenceLevel() == 5))) {
        // Level 5: LEGACY proofs also release the base zone early (post-commit-2) -- the
        // boundary-cost experiment; the conservative default keeps legacy release at setProof
        // because of the unfound Binary#1 boundary race.
        CHECKCUDAERR(cudaEventRecord(sdPh.baseFree, stream));
    }
    PROOFMAN_SUMCHECK("plain_zi_i%u", (gl64_t*)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("zi", true)]), (uint64_t)1 << setupCtx.starkInfo.starkStruct.nBitsExt, stream, (unsigned)instance_id);
    PROOFMAN_SUMCHECK("plain_preq1_i%u", (gl64_t*)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("cm1", true)]), ((uint64_t)1 << setupCtx.starkInfo.starkStruct.nBitsExt) * setupCtx.starkInfo.mapSectionsN["cm1"], stream, (unsigned)instance_id);
    PROOFMAN_SUMCHECK("plain_preq2_i%u", (gl64_t*)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("cm2", true)]), ((uint64_t)1 << setupCtx.starkInfo.starkStruct.nBitsExt) * setupCtx.starkInfo.mapSectionsN["cm2"], stream, (unsigned)instance_id);
    PROOFMAN_SUMCHECK("plain_preq3_i%u", (gl64_t*)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("cm3", true)]), ((uint64_t)1 << setupCtx.starkInfo.starkStruct.nBitsExt) * setupCtx.starkInfo.mapSectionsN["cm3"], stream, (unsigned)instance_id);
    PROOFMAN_SUMCHECK("plain_preqc_i%u", (gl64_t*)(h_params.pConstPolsExtendedTreeAddress), ((uint64_t)1 << setupCtx.starkInfo.starkStruct.nBitsExt) * setupCtx.starkInfo.nConstants, stream, (unsigned)instance_id);
    PROOFMAN_SUMCHECK("plain_preqav_i%u", (gl64_t*)h_params.airValues, setupCtx.starkInfo.airValuesSize, stream, (unsigned)instance_id);
    PROOFMAN_SUMCHECK("plain_preqagv_i%u", (gl64_t*)h_params.airgroupValues, setupCtx.starkInfo.airgroupValuesSize, stream, (unsigned)instance_id);
    PROOFMAN_SUMCHECK("plain_preqpub_i%u", (gl64_t*)h_params.publicInputs, setupCtx.starkInfo.nPublics, stream, (unsigned)instance_id);
    // Integrity check: the Q kernel dereferences the DEVICE StepsParams copy; verify it still
    // matches the host struct it was staged from (debug-only, PROOFMAN_SUMCHECK-gated).
    if (getenv("PROOFMAN_SUMCHECK") && instance_id == 1) {
        StepsParams dp_check;
        CHECKCUDAERR(cudaStreamSynchronize(stream));
        CHECKCUDAERR(cudaMemcpy(&dp_check, d_params, sizeof(StepsParams), cudaMemcpyDeviceToHost));
        fprintf(stderr, "[DPARAMS] i%u match=%d trace=%+ld aux=%+ld pub=%+ld ch=%+ld av=%+ld agv=%+ld cst=%+ld cstT=%+ld ccf=%+ld\n",
                (unsigned)instance_id, (int)(memcmp(&dp_check, &h_params, sizeof(StepsParams)) == 0),
                (long)((char*)dp_check.trace - (char*)h_params.trace),
                (long)((char*)dp_check.aux_trace - (char*)h_params.aux_trace),
                (long)((char*)dp_check.publicInputs - (char*)h_params.publicInputs),
                (long)((char*)dp_check.challenges - (char*)h_params.challenges),
                (long)((char*)dp_check.airValues - (char*)h_params.airValues),
                (long)((char*)dp_check.airgroupValues - (char*)h_params.airgroupValues),
                (long)((char*)dp_check.pConstPolsAddress - (char*)h_params.pConstPolsAddress),
                (long)((char*)dp_check.pConstPolsExtendedTreeAddress - (char*)h_params.pConstPolsExtendedTreeAddress),
                (long)((char*)dp_check.pCustomCommitsFixed - (char*)h_params.pCustomCommitsFixed));
    }
    cudagraph::run(cudagraph::key(0x51455850ULL ^ graphCtxId), countId, stream, [&] {
        calculateExpressionQ(setupCtx, air_instance_info->expressions_gpu, d_params, (Goldilocks::Element *)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]), d_expsArgs, d_destParams, pinned_exps_params, pinned_exps_args, countId, timer, stream, scratchShift);
    });
    PROOFMAN_SUMCHECK("tail_q_i%u", (gl64_t*)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]), ((uint64_t)1 << setupCtx.starkInfo.starkStruct.nBitsExt) * FIELD_EXTENSION, stream, (unsigned)instance_id);
    // Post-Q scratch head (pw table + first cross-chunk temps): bisects whether the generated
    // kernel's own table went wrong or the per-row chunk math.
    PROOFMAN_SUMCHECK("tail_scr_i%u", (gl64_t*)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("tmp1", false)]), (uint64_t)4096, stream, (unsigned)instance_id);
    // Debug-only raw dump of the q region for offline first-divergence localization
    // (PROOFMAN_DUMP_Q=<path-prefix>; dumps instance 1 only).
    if (const char *dq = getenv("PROOFMAN_DUMP_Q"); dq != nullptr && instance_id == 1) {
        uint64_t qWords = ((uint64_t)1 << setupCtx.starkInfo.starkStruct.nBitsExt) * FIELD_EXTENSION;
        uint64_t *hq = (uint64_t *)malloc(qWords * sizeof(uint64_t));
        CHECKCUDAERR(cudaMemcpyAsync(hq, h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)], qWords * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        CHECKCUDAERR(cudaStreamSynchronize(stream));
        char fn[512]; snprintf(fn, sizeof(fn), "%s_q_i%u_w%llu.bin", dq, (unsigned)instance_id, (unsigned long long)qWords);
        FILE *fp = fopen(fn, "wbx"); if (fp) { fwrite(hq, sizeof(uint64_t), qWords, fp); fclose(fp); }
        free(hq);
        // Also dump the generated kernel's scratch head (pw table + first temps).
        uint64_t sw = 4096;
        uint64_t *hs = (uint64_t *)malloc(sw * sizeof(uint64_t));
        CHECKCUDAERR(cudaMemcpy(hs, h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("tmp1", false)], sw * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        snprintf(fn, sizeof(fn), "%s_scr_i%u.bin", dq, (unsigned)instance_id);
        fp = fopen(fn, "wbx"); if (fp) { fwrite(hs, sizeof(uint64_t), sw, fp); fclose(fp); }
        free(hs);
    }
    TimerStopGPU(timer, STARK_QUOTIENT_POLYNOMIAL);
    cudagraph::run(cudagraph::key(0x434d5451ULL ^ graphCtxId), countId, stream, [&] {
        commitStage_inplace(setupCtx.starkInfo.nStages + 1, setupCtx, starks.treesGL, (gl64_t *)h_params.trace, (gl64_t *)h_params.aux_trace, d_transcript, false, timer, stream);
    });
    if (setupCtx.starkInfo.baseSplit && !recursive && splitPhases && splitAFenceLevel() == 2) CHECKCUDAERR(cudaEventRecord(sdPh.baseFree, stream));
    TimerStopGPU(timer, STARK_STEP_Q);
    TimerStartGPU(timer, STARK_STEP_EVALS);
    
    uint64_t xiChallengeIndex = 0;
    TimerStartCategoryGPU(timer, TRANSCRIPT);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2) {
            if(setupCtx.starkInfo.challengesMap[i].stageId == 0) xiChallengeIndex = i;
            d_transcript->getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    TimerStopCategoryGPU(timer, TRANSCRIPT);

    Goldilocks::Element *d_xiChallenge = &h_params.challenges[xiChallengeIndex * FIELD_EXTENSION];
    gl64_t * d_LEv = (gl64_t *) h_params.aux_trace +setupCtx.starkInfo.mapOffsets[std::make_pair("lev", false)];

    cudagraph::run(cudagraph::key(0x4556414cULL ^ graphCtxId), countId, stream, [&] {
    CHECKCUDAERR(cudaMemsetAsync(h_params.evals, 0, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element), stream));
    uint64_t count = 0;
    for(uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); i += EVALS_OPENING_BATCH) {
        std::vector<int64_t> openingPoints;
        for(uint64_t j = 0; j < EVALS_OPENING_BATCH; ++j) {
            if(i + j < setupCtx.starkInfo.openingPoints.size()) {
                openingPoints.push_back(setupCtx.starkInfo.openingPoints[i + j]);
            }
        }
        uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("lev_helper", false)];
        computeLEv_inplace(d_xiChallenge, setupCtx.starkInfo.starkStruct.nBits, openingPoints.size(), &air_instance_info->opening_points[i], d_aux_trace, offset_helper, d_LEv, timer, stream);
        evmap_inplace(setupCtx, h_params, count++, openingPoints.size(), openingPoints.data(), air_instance_info, (Goldilocks::Element*)d_LEv, offset_helper, timer, stream);
    }
    
    TimerStartCategoryGPU(timer, TRANSCRIPT);
    if(!setupCtx.starkInfo.starkStruct.hashCommits) {
        d_transcript->put(h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION, stream);
    } else {
        calculateHash(d_transcript_helper, d_challenge, setupCtx, h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION, stream);
        d_transcript->put(d_challenge, HASH_SIZE, stream);
    }

    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3) {
            d_transcript->getField((uint64_t *)&h_params.challenges[i * FIELD_EXTENSION], stream);
        }
    }
    TimerStopCategoryGPU(timer, TRANSCRIPT);
    });
    TimerStopGPU(timer, STARK_STEP_EVALS);
    if (setupCtx.starkInfo.baseSplit && !recursive && splitPhases && splitAFenceLevel() == 3) CHECKCUDAERR(cudaEventRecord(sdPh.baseFree, stream));
    PROOFMAN_SUMCHECK("tail_evals_i%u", (gl64_t*)h_params.evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION, stream, (unsigned)instance_id);
    PROOFMAN_SUMCHECK("tail_evch_i%u", (gl64_t*)h_params.challenges, setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION, stream, (unsigned)instance_id);
    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStartGPU(timer, STARK_STEP_FRI);
    // Outside the region, not in it: a replay skips the body, so a category timed inside one
    // stops being sampled. This category happens to cover the whole body, so it can just wrap it.
    TimerStartCategoryGPU(timer, FRI);
    cudagraph::run(cudagraph::key(0x46524950ULL ^ graphCtxId), countId, stream, [&] {
    calculateXis_inplace(setupCtx, h_params, air_instance_info->opening_points, d_xiChallenge, stream);
    uint64_t x_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("x", true)];
    dim3 threads(256);
    dim3 blocks((NExtended + threads.x - 1) / threads.x);
    computeX_kernel<<<blocks, threads, 0, stream>>>((gl64_t *)h_params.aux_trace + x_offset, NExtended, Goldilocks::shift(), Goldilocks::w(setupCtx.starkInfo.starkStruct.nBitsExt));
    });
    TimerStopCategoryGPU(timer, FRI);
    // Own capture region, as CALCULATE_IM_POLS: inside the FRIP body these two lost every replay.
    TimerStartGPU(timer, STARK_FRI_POLYNOMIAL);
    TimerStartCategoryGPU(timer, EXPRESSIONS);
    cudagraph::run(cudagraph::key(0x46524558ULL ^ graphCtxId), countId, stream, [&] {
        calculateFRIExpression(setupCtx, h_params, air_instance_info, stream);
    });
    TimerStopCategoryGPU(timer, EXPRESSIONS);
    TimerStopGPU(timer, STARK_FRI_POLYNOMIAL);
    for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) { 
        Goldilocks::Element *src = h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("fri_" + to_string(step + 1), true)];
        starks.treesFRI[step]->setSource(src);

        if(setupCtx.starkInfo.starkStruct.verificationHashType == "GL") {
            Goldilocks::Element *pBuffNodesGL = h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("mt_fri_" + to_string(step + 1), true)];
            starks.treesFRI[step]->setNodes(pBuffNodesGL);
        }
    }
    uint64_t friPol_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];
    uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("buff_helper", false)];
    gl64_t *d_friPol = (gl64_t *)(h_params.aux_trace + friPol_offset);
    
    uint64_t nBitsExt =  setupCtx.starkInfo.starkStruct.steps[0].nBits;

    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size(); step++)
    {
        uint64_t currentBits = setupCtx.starkInfo.starkStruct.steps[step].nBits;

        cudagraph::run(cudagraph::key(0x465249ULL ^ graphCtxId, step, nBitsExt, currentBits), countId, stream, [&] {
            if (step > 0) {
                uint64_t prevBits = setupCtx.starkInfo.starkStruct.steps[step - 1].nBits;
                fold_inplace(step, friPol_offset, offset_helper, d_challenge, nBitsExt, prevBits, currentBits, d_aux_trace, timer, stream);
            }
            if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
            {
                merkelizeFRI_inplace(setupCtx, h_params, step, d_friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, d_transcript, timer, stream);
            }
            else
            {
                if(!setupCtx.starkInfo.starkStruct.hashCommits) {
                    d_transcript->put((Goldilocks::Element *)d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION, stream);
                } else {
                    calculateHash(d_transcript_helper, d_challenge, setupCtx, (Goldilocks::Element *)d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION, stream);
                    d_transcript->put(d_challenge, HASH_SIZE, stream);
                }
            }
            d_transcript->getField((uint64_t *)d_challenge, stream);
        });
    }

    PROOFMAN_SUMCHECK("tail_f_i%u", (gl64_t*)(h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]), ((uint64_t)1 << setupCtx.starkInfo.starkStruct.nBitsExt) * FIELD_EXTENSION, stream, (unsigned)instance_id);
    TimerStartCategoryGPU(timer, GRINDING);
    Goldilocks::Element *d_input_hash_nonce = (Goldilocks::Element *)d_aux_trace + offsetInputHashNonce;
    CHECKCUDAERR(cudaMemcpyAsync(d_input_hash_nonce, d_challenge, FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToDevice, stream));
    cudagraph::run(cudagraph::key(0x4752494EULL ^ graphCtxId, setupCtx.starkInfo.starkStruct.powBits), countId, stream, [&] {
        runGrindingGPU((uint64_t *)d_nonce, (uint64_t *)d_nonceBlocks, (uint64_t *)d_input_hash_nonce, setupCtx.starkInfo.starkStruct.powBits, stream);
    });
    CHECKCUDAERR(cudaGetLastError());
    TimerStopCategoryGPU(timer, GRINDING);

    TimerStartCategoryGPU(timer, FRI);
    d_transcript_helper->reset(stream);
    d_transcript_helper->put2(d_challenge, FIELD_EXTENSION, d_nonce, 1, stream);
    Goldilocks::Element *permScratch = (Goldilocks::Element *)d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("fri_queries_perm", false)];
    d_transcript_helper->getPermutations(friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps[0].nBits, permScratch, stream);

    // Stage-1 overlap fence #2: the query phase is the first reader of the cm1 Merkle nodes.
    if (overlapStage1) {
        CHECKCUDAERR(cudaStreamWaitEvent(stream, sdOv.cm1TreeDone, 0));
    }
    // Query split: the FRI-tree openings go to the side stream (per-tree slices of
    // d_queries_buff are disjoint, the query indices are read-only, and the FRI trees were
    // built earlier on this stream so the fork event orders them). These are small
    // latency-bound kernels that under-occupy the GPU, so the two sets genuinely co-execute.
    if (overlapStage1) {
        CHECKCUDAERR(cudaEventRecord(sdOv.friQFork, stream));
        CHECKCUDAERR(cudaStreamWaitEvent(sdOv.sideStream, sdOv.friQFork, 0));
        // proveFRIQueries mutates its index buffer in place (moduleQueries folds it per step),
        // and the main stream's commitment-tree openings read the originals concurrently --
        // the side stream folds its own copy.
        uint64_t *friQueriesSide_gpu = (uint64_t *)d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("fri_queries_side", false)];
        CHECKCUDAERR(cudaMemcpyAsync(friQueriesSide_gpu, friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries * sizeof(uint64_t), cudaMemcpyDeviceToDevice, sdOv.sideStream));
        for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) {
            proveFRIQueries_inplace(setupCtx, &d_queries_buff[(nTrees + step) * setupCtx.starkInfo.starkStruct.nQueries * setupCtx.starkInfo.maxProofBuffSize], step + 1, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, friQueriesSide_gpu, setupCtx.starkInfo.starkStruct.nQueries, starks.treesFRI[step], sdOv.sideStream);
        }
        CHECKCUDAERR(cudaEventRecord(sdOv.friQDone, sdOv.sideStream));
    }
    // d_const_tree joins the key: preloaded const trees give the same air a different
    // (but per-stream-stable) tree pointer, and the query kernels read through it.
    cudagraph::run(cudagraph::key(0x515559ULL ^ graphCtxId ^ (uint64_t)(uintptr_t)d_const_tree, nTrees, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps.size()), countId, stream, [&] {
        proveQueries_inplace(setupCtx, d_queries_buff, friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, starks.treesGL, nTrees, d_aux_trace, d_const_tree, setupCtx.starkInfo.nStages, stream);
        if (!overlapStage1) {
        for(uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size() - 1; ++step) {
            proveFRIQueries_inplace(setupCtx, &d_queries_buff[(nTrees + step) * setupCtx.starkInfo.starkStruct.nQueries * setupCtx.starkInfo.maxProofBuffSize], step + 1, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits, friQueries_gpu, setupCtx.starkInfo.starkStruct.nQueries, starks.treesFRI[step], stream);
        }
        }
    });
    // setProof reads the whole d_queries_buff, including the side stream's FRI slices.
    if (overlapStage1) {
        CHECKCUDAERR(cudaStreamWaitEvent(stream, sdOv.friQDone, 0));
    }
    PROOFMAN_SUMCHECK("tail_queries_i%u", d_queries_buff, (nTrees + nTreesFRI) * setupCtx.starkInfo.maxProofBuffSize * setupCtx.starkInfo.starkStruct.nQueries, stream, (unsigned)instance_id);
    PROOFMAN_SUMCHECK("tail_nonce_i%u", (gl64_t*)d_nonce, 1, stream, (unsigned)instance_id);
    PROOFMAN_SUMCHECK("tail_mt1root_i%u", (gl64_t*)((Goldilocks::Element*)d_aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("mt1", true)] + setupCtx.starkInfo.getNumNodesMT(1ULL << setupCtx.starkInfo.starkStruct.nBitsExt) - HASH_SIZE), HASH_SIZE, stream, (unsigned)instance_id);
    TimerStopCategoryGPU(timer, FRI);
    TimerStopGPU(timer, STARK_STEP_FRI);

    if (setupCtx.starkInfo.baseSplit && !recursive && ((splitPhases && splitAFenceLevel() >= 4) || (!splitPhases && splitAFenceLevel() != 5))) CHECKCUDAERR(cudaEventRecord(sdPh.baseFree, stream));
    TimerStartCategoryGPU(timer, SET_PROOF);
    setProof(setupCtx, (Goldilocks::Element *)d_aux_trace, (Goldilocks::Element *)d_const_tree, proof_buffer_pinned, stream, smallsShift);
    TimerStopCategoryGPU(timer, SET_PROOF);

    TimerStopGPU(timer, STARK_GPU_PROOF);
}

#endif