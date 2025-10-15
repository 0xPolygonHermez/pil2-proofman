#ifndef GEN_COMMIT_CUH
#define GEN_COMMIT_CUH

#include "starks.hpp"
#include "cuda_utils.cuh"
#include "gl64_tooling.cuh"
#include "starks_gpu.cuh"


void genCommit_gpu(uint64_t arity, uint64_t nBits, uint64_t nBitsExtended, uint64_t nCols, gl64_t *d_aux_trace, Goldilocks::Element *root_pinned, SetupCtx *setupCtx, AirInstanceInfo *air_instance_info, TimerGPU &timer, cudaStream_t stream, uint64_t nStreams = 1) {
    TimerStartGPU(timer, STARK_GPU_COMMIT);
    uint64_t N = 1 << nBits;
    uint64_t NExtended = 1 << nBitsExtended;
    if (nCols > 0)
    {
        gl64_t *src = d_aux_trace;
        gl64_t *dst = d_aux_trace;

        uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(NExtended, arity);

        uint64_t offset_src = nStreams == 1  ? setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", false)] : 0;
        uint64_t offset_dst = nStreams == 1  ? setupCtx->starkInfo.mapOffsets[std::make_pair("cm1", true)] : N * nCols;
        uint64_t offset_mt = nStreams == 1  ? setupCtx->starkInfo.mapOffsets[make_pair("mt1", true)] : (N + NExtended) * nCols;

        Goldilocks::Element *pNodes = (Goldilocks::Element*)dst + offset_mt;
        NTT_Goldilocks_GPU ntt;

        if (air_instance_info->is_packed) {
            unpack_trace(air_instance_info, (uint64_t *)(src + offset_dst), (uint64_t *)(src + offset_src), nCols, N, stream, timer);
        } else {
            ntt.prepare_blocks_trace((gl64_t *)(src + offset_src), (gl64_t *)(src + offset_dst), nCols, N, stream, timer);
        }
        
        ntt.LDE_MerkleTree_GPU_inplace(pNodes, dst, offset_dst, src, offset_src, nBits, nBitsExtended, nCols, timer, stream);
        CHECKCUDAERR(cudaMemcpyAsync(root_pinned, &pNodes[tree_size - HASH_SIZE], HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    } else {
        std::cout << "nCols must be greater than 0" << std::endl;
        assert(0);
    }
    TimerStopGPU(timer, STARK_GPU_COMMIT);
}

#endif