#ifndef GEN_COMMIT_CUH
#define GEN_COMMIT_CUH

#include "starks.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "starks_gpu.cuh"


void genCommit_gpu(uint64_t arity, uint64_t nBits, uint64_t nBitsExtended, uint64_t nCols, gl64_t *d_aux_trace, Goldilocks::Element *root_pinned, TimerGPU &timer, cudaStream_t stream) {

    TimerStartGPU(timer, GEN_COMMIT_GPU);

    uint64_t NExtended = 1 << nBitsExtended;
    if (nCols > 0)
    {
        gl64_t *src = d_aux_trace;
        uint64_t offset_src = 0;
        gl64_t *dst = d_aux_trace;
        uint64_t offset_dst = 0;

        uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(NExtended, arity);

        NTT_Goldilocks_GPU ntt;

        uint64_t offset_aux = NExtended * nCols + tree_size;

        Goldilocks::Element *pNodes = (Goldilocks::Element*) d_aux_trace + nCols * NExtended;
        ntt.LDE_MerkleTree_GPU_inplace(pNodes, dst, offset_dst, src, offset_src, nBits, nBitsExtended, nCols, d_aux_trace, offset_aux, timer, stream);
        CHECKCUDAERR(cudaMemcpyAsync(root_pinned, pNodes + tree_size - HASH_SIZE, HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    } else {
        std::cout << "nCols must be greater than 0" << std::endl;
        assert(0);
    }

    TimerStopGPU(timer, GEN_COMMIT_GPU);
}

#endif