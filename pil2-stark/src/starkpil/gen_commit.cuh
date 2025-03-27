#ifndef GEN_COMMIT_CUH
#define GEN_COMMIT_CUH

#include "starks.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "starks_gpu.cuh"

void genCommit_gpu(uint64_t arity, Goldilocks::Element* root, uint64_t N, uint64_t NExtended, uint64_t nCols, DeviceCommitBuffers *d_buffers) {

    CHECKCUDAERR(cudaDeviceSynchronize());
    double time = omp_get_wtime();

    if (nCols > 0)
    {
        gl64_t *src = d_buffers->d_trace;
        uint64_t offset_src = 0;
        gl64_t *dst = d_buffers->d_aux_trace;
        uint64_t offset_dst = 0;

        NTT_Goldilocks ntt(N);

        
        Goldilocks::Element *pNodes = (Goldilocks::Element*) d_buffers->d_aux_trace + nCols * NExtended;
        uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements(NExtended, 3);
        ntt.LDE_MerkleTree_GPU_inplace(pNodes, dst, offset_dst, src, offset_src, N, NExtended, nCols, d_buffers);
        CHECKCUDAERR(cudaMemcpy(&root[0], pNodes + tree_size - HASH_SIZE, HASH_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost));   
    } else {
        std::cout << "nCols must be greater than 0" << std::endl;
        assert(0);
    }

    CHECKCUDAERR(cudaDeviceSynchronize());
    time = omp_get_wtime() - time;
    //std::cout << "genCommit_gpu " <<time<<std::endl;

}

#endif