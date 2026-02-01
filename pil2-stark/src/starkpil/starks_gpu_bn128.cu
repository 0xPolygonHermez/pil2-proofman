#include "starks_gpu_bn128.cuh"
#include "transcript/transcriptBN128.cuh"
#include "setup_ctx.hpp"
#include "ntt_goldilocks.cuh"
#include "starks_gpu.cuh"

class gl64_t;

__global__ void convertGLToBN128ScalarField_kernel(
    PoseidonBN128GPU::FrElement *output,
    const uint64_t *input,
    uint64_t n
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Create element from uint64 using the Element operator[]
        uint64_t gl_val = input[idx];
        output[idx][0] = (uint32_t)gl_val;
        output[idx][1] = (uint32_t)(gl_val >> 32);
        output[idx][2] = 0;
        output[idx][3] = 0;
        output[idx][4] = 0;
        output[idx][5] = 0;
        output[idx][6] = 0;
        output[idx][7] = 0;
        
        // Convert to Montgomery form
        BN128GPUScalarField::toMontgomery(output[idx]);
    }
}

void convertGLToBN128ScalarField(PoseidonBN128GPU::FrElement *output, const uint64_t *input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return;
    dim3 threads(32);
    dim3 blocks((n + threads.x - 1) / threads.x);
    convertGLToBN128ScalarField_kernel<<<blocks, threads, 0, stream>>>(output, input, n);
}

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
        computeQ_bn128_gpu(step, setupCtx, treesGL, d_aux_trace, d_transcript, timer, stream);
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
        PoseidonBN128GPU::merkletreeTiles(pNodes, (uint64_t*)pSource, nCols, NExtended, setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, stream);
        if(d_transcript != nullptr) {
            d_transcript->put(&pNodes[tree_size - 1], 1, stream);
        }
    } 
    // Note: pNodes is stored in treesGL[step-1] via setNodes() and will be freed when treesGL is destroyed
}

void computeQ_bn128_gpu(uint64_t step, SetupCtx &setupCtx, MerkleTreeBN128 **treesGL, Goldilocks::Element *d_aux_trace,TranscriptBN128_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream)
{
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    std::string section = "cm" + to_string(step);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];

    uint64_t offset_cmQ = setupCtx.starkInfo.mapOffsets[std::make_pair(section, true)];
    uint64_t offset_q = setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)];
    uint64_t qDeg = setupCtx.starkInfo.qDeg;
    uint64_t qDim = setupCtx.starkInfo.qDim;

    Goldilocks::Element shiftIn = Goldilocks::exp(Goldilocks::inv(Goldilocks::shift()), N);
     
    Goldilocks::Element *pSource = d_aux_trace + offset_cmQ;
    treesGL[step - 1]->setSource(pSource);
    PoseidonBN128GPU::FrElement * pNodes;
    int64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended);
    cudaMalloc((void**)&pNodes, tree_size * sizeof(PoseidonBN128GPU::FrElement));
    treesGL[step - 1]->setNodes((RawFr::Element*)pNodes);

    if (nCols > 0)
    {
        uint64_t offset_helper = setupCtx.starkInfo.mapOffsets[std::make_pair("extra_helper_fft", false)];
        NTT_Goldilocks_GPU nttExtended;
        nttExtended.computeQ_inplace(offset_cmQ, offset_q, qDeg, qDim, shiftIn, setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.starkStruct.nBitsExt, nCols, (gl64_t*)d_aux_trace, offset_helper, timer, stream);
        PoseidonBN128GPU::merkletreeTiles(pNodes, (uint64_t*)pSource, nCols, NExtended, setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, stream);
        if(d_transcript != nullptr) {
            d_transcript->put(&pNodes[tree_size - 1], 1, stream);
        }
    }
    // Note: pNodes is stored in treesGL[step-1] via setNodes() and will be freed when treesGL is destroyed
}

void merkelizeFRI_bn128_gpu(SetupCtx& setupCtx, StepsParams &h_params, uint64_t step, Goldilocks::Element *pol, MerkleTreeBN128 *treeFRI, uint64_t currentBits, uint64_t nextBits, TranscriptBN128_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream)
{
    uint64_t pol2N = 1 << currentBits;

    uint64_t width = 1 << nextBits;
    uint64_t height = pol2N / width;
    dim3 nThreads(32, 32);
    dim3 nBlocks((width + nThreads.x - 1) / nThreads.x, (height + nThreads.y - 1) / nThreads.y);

    Goldilocks::Element *src = h_params.aux_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("fri_" + to_string(step + 1), true)];
    treeFRI->setSource(src); 
    transposeFRI<<<nBlocks, nThreads, 0, stream>>>((gl64_t *)treeFRI->source, (gl64_t *)pol, pol2N, width);
    
    TimerStartCategoryGPU(timer, MERKLE_TREE);
    PoseidonBN128GPU::FrElement * pNodes;
    int64_t tree_size = treeFRI->getNumNodes(treeFRI->height);
    cudaMalloc((void**)&pNodes, tree_size * sizeof(PoseidonBN128GPU::FrElement));
    treeFRI->setNodes((RawFr::Element*)pNodes);
    PoseidonBN128GPU::merkletree((PoseidonBN128GPU::FrElement*) treeFRI->nodes, (uint64_t *)treeFRI->source, treeFRI->width, treeFRI->height, setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom, stream);
    
    TimerStopCategoryGPU(timer, MERKLE_TREE);

    if(d_transcript != nullptr) {
        d_transcript->put((PoseidonBN128GPU::FrElement*)&treeFRI->nodes[tree_size - 1], uint64_t(1), stream);
    }
}