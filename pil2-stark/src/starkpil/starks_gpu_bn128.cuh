#ifndef STARKS_GPU_BN128_CUH
#define STARKS_GPU_BN128_CUH

#include "bn128.cuh"
#include "poseidon_bn128.cuh"
#include "setup_ctx.hpp"
#include "transcriptBN128.cuh"
#include "gpu_timer.cuh"
#include "steps.hpp"
#include "gl64_t.cuh"
#include "proof_stark.hpp"
#include "starks.hpp"


void calculateHashBN128_gpu(TranscriptBN128_GPU *d_transcript ,PoseidonBN128GPU::FrElement* hash, SetupCtx &setupCtx, Goldilocks::Element* buffer, uint64_t nElements, cudaStream_t stream);

void convertGLToBN128ScalarField(PoseidonBN128GPU::FrElement *output, const uint64_t *input, uint64_t n, cudaStream_t stream);

void commitStage_bn128_gpu(uint64_t step, SetupCtx& setupCtx, MerkleTreeBN128**treesGL, Goldilocks::Element* d_witness, Goldilocks::Element* d_aux_trace, TranscriptBN128_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream);

void extendAndMerkelize_bn128_gpu(uint64_t step, SetupCtx& setupCtx, MerkleTreeBN128** treesGL, Goldilocks::Element* d_trace, Goldilocks::Element* d_aux_trace, TranscriptBN128_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream);

void computeQ_bn128_gpu(uint64_t step, SetupCtx& setupCtx, MerkleTreeBN128 **treesGL, Goldilocks::Element *d_aux_trace, TranscriptBN128_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream);

void merkelizeFRI_bn128_gpu(SetupCtx& setupCtx, StepsParams &h_params, uint64_t step, Goldilocks::Element *pol, MerkleTreeBN128 *treeFRI, uint64_t currentBits, uint64_t nextBits, TranscriptBN128_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream);

void proveQueries_bn128_gpu(SetupCtx& setupCtx, gl64_t *d_queries_buff, uint64_t *d_friQueries, uint64_t nQueries, MerkleTreeBN128 **trees, uint64_t nTrees, gl64_t *d_aux_trace, uint32_t nStages, cudaStream_t stream);

void proveFRIQueries_bn128_gpu(SetupCtx& setupCtx, gl64_t *d_queries_buff, uint64_t step, uint64_t currentBits, uint64_t *d_friQueries, uint64_t nQueries, MerkleTreeBN128 *treeFRI, cudaStream_t stream);

void setProof_bn128_gpu(
    Starks<RawFr::Element>& starks,
    FRIProof<RawFr::Element>& proof,
    Goldilocks::Element *d_aux_trace,
    cudaStream_t stream
);

#endif