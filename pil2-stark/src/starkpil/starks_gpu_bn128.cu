#include "starks_gpu_bn128.cuh"
#include "transcript/transcriptBN128.cuh"
#include "setup_ctx.hpp"
#include "ntt_goldilocks.cuh"


void calculateHashBN128_gpu(TranscriptBN128_GPU *d_transcript, PoseidonBN128GPU::FrElement* hash, SetupCtx &setupCtx, Goldilocks::Element* buffer, uint64_t nElements, cudaStream_t stream) {

    d_transcript->reset(stream);
    d_transcript->put(buffer, nElements, stream);
    d_transcript->getState(hash, stream);
}

void commitStage_bn128_gpu(uint64_t step, SetupCtx &setupCtx, MerkleTreeBN128 **treesGL, Goldilocks::Element* d_trace, Goldilocks::Element*d_aux_trace, TranscriptBN128_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream)
{
    if (step <= setupCtx.starkInfo.nStages)
    {
        extendAndMerkelize_bn128_gpu(step, setupCtx, treesGL, d_trace, d_aux_trace, d_transcript, timer, stream);
    }
    else
    {
        //computeQ_inplace(step, setupCtx, treesGL, d_aux_trace, d_transcript, timer, stream);
    }
}

void extendAndMerkelize_bn128_gpu(uint64_t step, SetupCtx& setupCtx, MerkleTreeBN128** treesGL, Goldilocks::Element* d_trace, Goldilocks::Element* d_aux_trace, TranscriptBN128_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream)
{
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    std::string section = "cm" + to_string(step);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];

    gl64_t *src = step == 1 ? (gl64_t*) d_trace : (gl64_t*) d_aux_trace;
    uint64_t offset_src = step == 1 ? 0 : setupCtx.starkInfo.mapOffsets[make_pair(section, false)];
    gl64_t *dst = (gl64_t*) d_aux_trace;
    uint64_t offset_dst = setupCtx.starkInfo.mapOffsets[make_pair(section, true)];
    
    // Debug: print input witness first values
    if (step == 1) {
        std::vector<uint64_t> h_input(16);
        cudaMemcpy(h_input.data(), src + offset_src, 16 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        std::cout << "=== GPU Stage 1 Input (first 16 values) ===" << std::endl;
        for (int i = 0; i < 16; i++) {
            std::cout << "input[" << i << "]: " << h_input[i] << std::endl;
        }
        std::cout << "N=" << N << " NExtended=" << NExtended << " nCols=" << nCols << std::endl;
        std::cout << "============================================" << std::endl;
    }
    
    Goldilocks::Element *pSource = d_aux_trace + offset_dst;
    treesGL[step - 1]->setSource(pSource);
    PoseidonBN128GPU::FrElement * pNodes;
    int64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
    cudaMalloc((void**)&pNodes, tree_size * sizeof(PoseidonBN128GPU::FrElement));
    treesGL[step - 1]->setNodes((RawFr::Element*)pNodes);


    if (nCols > 0)
    {
        NTT_Goldilocks_GPU ntt;
        ntt.LDE_GPU(dst, offset_dst, src, offset_src, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, nCols, timer, stream);
        // Use pSource which already has offset_dst applied (d_aux_trace + offset_dst)
        PoseidonBN128GPU::merkletreeTiles(pNodes, (uint64_t*)pSource, nCols, NExtended, setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, stream);
        if(d_transcript != nullptr) {
            d_transcript->put(&pNodes[tree_size - 1], 1, stream);
        }
    } 
    // Note: pNodes is stored in treesGL[step-1] via setNodes() and will be freed when treesGL is destroyed
}