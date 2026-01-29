#ifndef STARKS_GPU_BN128_CUH
#define STARKS_GPU_BN128_CUH

#include "bn128.cuh"
#include "poseidon_bn128.cuh"
#include "setup_ctx.hpp"
#include "transcriptBN128.cuh"
#include "gpu_timer.cuh"


void calculateHashBN128_gpu(TranscriptBN128_GPU *d_transcript ,PoseidonBN128GPU::FrElement* hash, SetupCtx &setupCtx, Goldilocks::Element* buffer, uint64_t nElements, cudaStream_t stream);

void commitStage_bn128_gpu(uint64_t step, SetupCtx& setupCtx, MerkleTreeBN128**treesGL, Goldilocks::Element* d_witness, Goldilocks::Element* d_aux_trace, TranscriptBN128_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream);

void extendAndMerkelize_bn128_gpu(uint64_t step, SetupCtx& setupCtx, MerkleTreeBN128** treesGL, Goldilocks::Element* d_trace, Goldilocks::Element* d_aux_trace, TranscriptBN128_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream);

void computeQ_bn128_gpu(uint64_t step, SetupCtx& setupCtx, MerkleTreeBN128 **treesGL, Goldilocks::Element *d_aux_trace, TranscriptBN128_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream);


#endif