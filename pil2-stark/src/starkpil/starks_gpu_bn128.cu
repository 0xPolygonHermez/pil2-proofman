#include "starks_gpu_bn128.cuh"
#include "transcript/transcriptBN128.cuh"
#include "setup_ctx.hpp"
#include "ntt_goldilocks.cuh"
#include "starks_gpu.cuh"
#include "data_layout.cuh"

class gl64_t;

// Goldilocks modulus for reduction
#define GOLDILOCKS_PRIME 0xFFFFFFFF00000001ULL

// Kernel to extract polynomial values at queried positions from trace
__global__ void getTracePolsTilesBN128(gl64_t *d_treeTrace, uint64_t nCols, uint64_t nRows, uint64_t *d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t querySize)
{
    uint64_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x < nCols && idx_y < nQueries)
    {
        uint64_t row = d_friQueries[idx_y];
        uint64_t idx_buffer = idx_y * querySize + idx_x;
        // Use the proper tiled format from data_layout.cuh
        uint64_t idx_trace = getBufferOffset(row, idx_x, nRows, nCols);
        uint64_t val = d_treeTrace[idx_trace][0];
        // Reduce the Goldilocks value
        if (val >= GOLDILOCKS_PRIME) {
            val -= GOLDILOCKS_PRIME;
        }
        d_buffer[idx_buffer] = gl64_t(val);
    }
}

// Device function to recursively generate Merkle proof for BN128 trees
// Writes BN128 siblings (RawFr::Element = 8 x uint32_t) to proof buffer
__device__ void genMerkleProofBN128_(
    PoseidonBN128GPU::FrElement *nodes,
    PoseidonBN128GPU::FrElement *proof,
    uint64_t idx,
    uint64_t offset,
    uint64_t n,
    uint32_t arity,
    uint64_t lastLevel
) {
    if ((lastLevel == 0 && n == 1) || (lastLevel > 0 && (n <= pow((double)arity, (double)lastLevel)))) return;

    uint64_t currIdx = idx % arity;
    uint64_t nextIdx = idx / arity;
    uint64_t si = idx - currIdx;  // start index of the group

    // Copy ALL arity siblings (including the one at currIdx, matching CPU behavior)
    for (uint64_t i = 0; i < arity; i++)
    {
        // Copy the BN128 element (8 x uint32_t)
        for (uint32_t j = 0; j < 8; j++) {
            proof[i][j] = nodes[offset + si + i][j];
        }
    }

    uint64_t extraZeros = (arity - (n % arity)) % arity;
    uint64_t nPadded = n + extraZeros;
    uint64_t nextN = nPadded / arity;
    
    genMerkleProofBN128_(nodes, &proof[arity], nextIdx, offset + nPadded, nextN, arity, lastLevel);
}

// Kernel to generate Merkle proofs for BN128 trees
// Each thread handles one query
__global__ void genMerkleProofBN128(
    PoseidonBN128GPU::FrElement *d_nodes,
    uint64_t nLeaves,
    uint64_t *d_friQueries,
    uint64_t nQueries,
    gl64_t *d_buffer,           // Output buffer (mixed: GL polynomial values + BN128 proofs)
    uint64_t bufferWidth,       
    uint64_t maxTreeWidth,      
    uint64_t arity,
    uint64_t lastLevel
) {
    uint64_t idx_query = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_query < nQueries)
    {
        uint64_t row = d_friQueries[idx_query];
        uint8_t *bufferBase = (uint8_t*)d_buffer;
        uint8_t *proofStart = bufferBase + idx_query * bufferWidth * sizeof(gl64_t) + maxTreeWidth * sizeof(gl64_t);
        PoseidonBN128GPU::FrElement *proof = (PoseidonBN128GPU::FrElement *)proofStart; 
        
        genMerkleProofBN128_(d_nodes, proof, row, 0, nLeaves, arity, lastLevel);
    }
}

// Kernel to reduce query indices modulo current FRI domain size
__global__ void moduleQueriesBN128(uint64_t* d_friQueries, uint64_t nQueries, uint64_t currentBits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nQueries) {
        d_friQueries[idx] %= (1ULL << currentBits);
    }
}

// Kernel to extract FRI polynomial values at queried positions
// FRI data is stored after transposeFRI in ROW-MAJOR format: leaf * width + col
__global__ void getTracePolsFRIBN128(gl64_t *d_treeTrace, uint64_t nCols, uint64_t *d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth)
{
    uint64_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x < nCols && idx_y < nQueries)
    {
        uint64_t row = d_friQueries[idx_y];
        uint64_t idx_buffer = idx_y * bufferWidth + idx_x;
        uint64_t idx_trace = row * nCols + idx_x;
        uint64_t val = d_treeTrace[idx_trace][0];
        // Reduce the Goldilocks value
        if (val >= GOLDILOCKS_PRIME) {
            val -= GOLDILOCKS_PRIME;
        }
        d_buffer[idx_buffer] = gl64_t(val);
    }
}

void proveQueries_bn128_gpu(
    SetupCtx& setupCtx,
    gl64_t *d_queries_buff,
    uint64_t *d_friQueries,
    uint64_t nQueries,
    MerkleTreeBN128 **trees,
    uint64_t nTrees,
    gl64_t *d_aux_trace,
    uint32_t nStages,
    cudaStream_t stream
) {
    uint64_t maxBuffSize = setupCtx.starkInfo.maxProofBuffSize;
    uint64_t maxTreeWidth = setupCtx.starkInfo.maxTreeWidth;

    // For each tree, extract polynomial values and generate Merkle proofs
    for (uint64_t k = 0; k < nTrees; k++)
    {
        // Extract polynomial values at queried positions
        dim3 nThreads(32, 32);
        dim3 nBlocks((trees[k]->getMerkleTreeWidth() + nThreads.x - 1) / nThreads.x, (nQueries + nThreads.y - 1) / nThreads.y);
        
        getTracePolsTilesBN128<<<nBlocks, nThreads, 0, stream>>>(
            (gl64_t*)trees[k]->source,
            trees[k]->getMerkleTreeWidth(),
            trees[k]->height,
            d_friQueries,
            nQueries,
            d_queries_buff + k * nQueries * maxBuffSize,
            maxBuffSize
        );
        CHECKCUDAERR(cudaGetLastError());

        // Generate Merkle proofs (BN128 siblings)
        dim3 nthreads(64);
        dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
        genMerkleProofBN128<<<nblocks, nthreads, 0, stream>>>(
            (PoseidonBN128GPU::FrElement *)trees[k]->nodes,
            trees[k]->height,
            d_friQueries,
            nQueries,
            d_queries_buff + k * nQueries * maxBuffSize,
            maxBuffSize,
            maxTreeWidth,
            setupCtx.starkInfo.starkStruct.merkleTreeArity,
            setupCtx.starkInfo.starkStruct.lastLevelVerification
        );
    }
    CHECKCUDAERR(cudaGetLastError());
}

void proveFRIQueries_bn128_gpu(
    SetupCtx& setupCtx,
    gl64_t *d_queries_buff,
    uint64_t step,
    uint64_t currentBits,
    uint64_t *d_friQueries,
    uint64_t nQueries,
    MerkleTreeBN128 *treeFRI,
    cudaStream_t stream
) {
    uint64_t buffSize = treeFRI->getMerkleTreeWidth() + treeFRI->getMerkleProofSize() / sizeof(gl64_t);
    
    // Reduce query indices modulo current domain size
    dim3 nthreads_(64);
    dim3 nblocks_((nQueries + nthreads_.x - 1) / nthreads_.x);
    moduleQueriesBN128<<<nblocks_, nthreads_, 0, stream>>>(d_friQueries, nQueries, currentBits);
    CHECKCUDAERR(cudaGetLastError());

    // Extract polynomial values at queried positions
    dim3 nThreads(32, 32);
    dim3 nBlocks((treeFRI->getMerkleTreeWidth() + nThreads.x - 1) / nThreads.x, (nQueries + nThreads.y - 1) / nThreads.y);
    getTracePolsFRIBN128<<<nBlocks, nThreads, 0, stream>>>(
        (gl64_t *)treeFRI->source,
        treeFRI->getMerkleTreeWidth(),
        d_friQueries,
        nQueries,
        d_queries_buff,
        buffSize
    );
    CHECKCUDAERR(cudaGetLastError());

    // Generate Merkle proofs
    dim3 nthreads(64);
    dim3 nblocks((nQueries + nthreads.x - 1) / nthreads.x);
    genMerkleProofBN128<<<nblocks, nthreads, 0, stream>>>(
        (PoseidonBN128GPU::FrElement *)treeFRI->nodes,
        treeFRI->height,
        d_friQueries,
        nQueries,
        d_queries_buff,
        buffSize,
        treeFRI->getMerkleTreeWidth(),
        setupCtx.starkInfo.starkStruct.merkleTreeArity,
        setupCtx.starkInfo.starkStruct.lastLevelVerification
    );
    CHECKCUDAERR(cudaGetLastError());
}

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