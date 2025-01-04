#ifndef GEN_RECURSIVE_PROOF_GPU_HPP
#define GEN_RECURSIVE_PROOF_GPU_HPP

#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "cuda_utils.cuh"
#include "gl64_t.cuh"
#include "expressions_gpu.cuh"


Goldilocks::Element omegas_inv_[33] = {
    0x1,
    0xffffffff00000000,
    0xfffeffff00000001,
    0xfffffeff00000101,
    0xffefffff00100001,
    0xfbffffff04000001,
    0xdfffffff20000001,
    0x3fffbfffc0,
    0x7f4949dce07bf05d,
    0x4bd6bb172e15d48c,
    0x38bc97652b54c741,
    0x553a9b711648c890,
    0x55da9bb68958caa,
    0xa0a62f8f0bb8e2b6,
    0x276fd7ae450aee4b,
    0x7b687b64f5de658f,
    0x7de5776cbda187e9,
    0xd2199b156a6f3b06,
    0xd01c8acd8ea0e8c0,
    0x4f38b2439950a4cf,
    0x5987c395dd5dfdcf,
    0x46cf3d56125452b1,
    0x909c4b1a44a69ccb,
    0xc188678a32a54199,
    0xf3650f9ddfcaffa8,
    0xe8ef0e3e40a92655,
    0x7c8abec072bb46a6,
    0xe0bfc17d5c5a7a04,
    0x4c6b8a5a0b79f23a,
    0x6b4d20533ce584fe,
    0xe5cceae468a70ec2,
    0x8958579f296dac7a,
    0x16d265893b5b7e85,
};

__global__ void insertTracePol(Goldilocks::Element* d_trace, uint64_t offset, uint64_t stride, Goldilocks::Element* d_pol, uint64_t dim, uint64_t N){
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        if(dim == 1) d_trace[offset + idx * stride] = d_pol[idx];
        else{
            d_trace[offset + idx * stride] = d_pol[idx * dim ];
            d_trace[offset + idx * stride + 1] = d_pol[idx * dim + 1];
            d_trace[offset + idx * stride + 2] = d_pol[idx * dim + 2];
        }
    }
}

__device__ __constant__ uint64_t domain_size_inverse_[33] = {
    0x0000000000000001, // 1^{-1}
    0x7fffffff80000001, // 2^{-1}
    0xbfffffff40000001, // (1 << 2)^{-1}
    0xdfffffff20000001, // (1 << 3)^{-1}
    0xefffffff10000001,
    0xf7ffffff08000001,
    0xfbffffff04000001,
    0xfdffffff02000001,
    0xfeffffff01000001,
    0xff7fffff00800001,
    0xffbfffff00400001,
    0xffdfffff00200001,
    0xffefffff00100001,
    0xfff7ffff00080001,
    0xfffbffff00040001,
    0xfffdffff00020001,
    0xfffeffff00010001,
    0xffff7fff00008001,
    0xffffbfff00004001,
    0xffffdfff00002001,
    0xffffefff00001001,
    0xfffff7ff00000801,
    0xfffffbff00000401,
    0xfffffdff00000201,
    0xfffffeff00000101,
    0xffffff7f00000081,
    0xffffffbf00000041,
    0xffffffdf00000021,
    0xffffffef00000011,
    0xfffffff700000009,
    0xfffffffb00000005,
    0xfffffffd00000003,
    0xfffffffe00000002, // (1 << 32)^{-1}
};

void offloadCommit(uint64_t step, MerkleTreeGL** treesGL, Goldilocks::Element *trace, gl64_t *d_trace, uint64_t* d_tree, FRIProof<Goldilocks::Element> &proof, SetupCtx& setupCtx){

    uint64_t ncols = setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)];
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    //uint64_t size = NExtended * ncols * sizeof(Goldilocks::Element);
    uint64_t tree_size = treesGL[step - 1]->getNumNodes(NExtended) * sizeof(uint64_t);
    std::string section = "cm" + to_string(step);  
    uint64_t offset = setupCtx.starkInfo.mapOffsets[make_pair(section, true)];
    treesGL[step - 1]->setSource(trace + offset);
    treesGL[step - 1]->souceTraceOffset=offset;
    if(ncols > 0){
        //CHECKCUDAERR(cudaMemcpy(trace + offset, d_trace + offset, size, cudaMemcpyDeviceToHost));
        CHECKCUDAERR(cudaMemcpy(treesGL[step - 1]->get_nodes_ptr(), d_tree, tree_size, cudaMemcpyDeviceToHost));
    }else{
        treesGL[step - 1]->merkelize();
    }
    treesGL[step - 1]->getRoot(&proof.proof.roots[step - 1][0]);
}

__global__ void fillLEv(uint64_t LEv_offset, gl64_t * d_xiChallenge, uint64_t W_, uint64_t nOpeningPoints, int64_t *d_openingPoints, uint64_t shift_, gl64_t * d_trace, uint64_t N){
    
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < nOpeningPoints){
        gl64_t w(1);
        Goldilocks3GPU::Element xi;
        Goldilocks3GPU::Element xiShifted;
        uint64_t openingAbs = d_openingPoints[i] < 0 ? -d_openingPoints[i] : d_openingPoints[i];
        gl64_t * LEv = (gl64_t *) d_trace + LEv_offset;
        gl64_t W(W_);
        gl64_t shift(shift_);
        gl64_t invShift = shift.reciprocal();
        for (uint64_t j = 0; j < openingAbs; ++j)
        {
            w *=  W;
        }

        if (d_openingPoints[i] < 0)
        {
            w = w.reciprocal();
        }
        Goldilocks3GPU::mul(xi, *((Goldilocks3GPU::Element *)d_xiChallenge), w);
        Goldilocks3GPU::mul(xiShifted, xi, invShift);
        Goldilocks3GPU::one((*(Goldilocks3GPU::Element *) &LEv[i * FIELD_EXTENSION]));
        
        for (uint64_t k = 1; k < N; k++)
        {
            Goldilocks3GPU::mul((*(Goldilocks3GPU::Element *) &LEv[(k*nOpeningPoints + i)*FIELD_EXTENSION]), (*(Goldilocks3GPU::Element *) &LEv[((k-1)*nOpeningPoints + i)*FIELD_EXTENSION]), xiShifted);
        }
    }
}

void computeLEv_inplace(Goldilocks::Element *xiChallenge, uint64_t LEv_offset, uint64_t nBits, uint64_t nOpeningPoints,int64_t *openingPoints, DeviceCommitBuffers* d_buffers){

    double time = omp_get_wtime();
    uint64_t N = 1 << nBits;

    gl64_t * d_xiChallenge;
    int64_t * d_openingPoints;
    cudaMalloc(&d_xiChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element));
    cudaMemcpy(d_xiChallenge, xiChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    cudaMalloc(&d_openingPoints, nOpeningPoints * sizeof(int64_t));
    cudaMemcpy(d_openingPoints, openingPoints, nOpeningPoints * sizeof(int64_t), cudaMemcpyHostToDevice);

    dim3 nThreads(32);
    dim3 nBlocks((nOpeningPoints + nThreads.x - 1) / nThreads.x);
    fillLEv<<<nBlocks, nThreads>>>(LEv_offset, d_xiChallenge, Goldilocks::w(nBits).fe, nOpeningPoints, d_openingPoints, Goldilocks::shift().fe, d_buffers->d_trace, N);
    time = omp_get_wtime() - time;
    std::cout << "LEv inplace: " << time << std::endl;
    
    time = omp_get_wtime();
    NTT_Goldilocks ntt(N);
    ntt.INTT_inplace(LEv_offset, N, FIELD_EXTENSION * nOpeningPoints, d_buffers);
    time = omp_get_wtime() - time;
    std::cout << "INTT: " << time << std::endl;

}

__global__ void calcXDivXSub(uint64_t xDivXSub_offset, gl64_t *d_xiChallenge, uint64_t W_, uint64_t nOpeningPoints, int64_t *d_openingPoints, gl64_t * d_x, gl64_t * d_trace, uint64_t NExtended){
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t k = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < nOpeningPoints){
        Goldilocks3GPU::Element xi;
        gl64_t w(1);
        uint64_t openingAbs = d_openingPoints[i] < 0 ? -d_openingPoints[i] : d_openingPoints[i];
        gl64_t W(W_);
        for (uint64_t j = 0; j < openingAbs; ++j)
        {
            w *=  W;
        }
        if (d_openingPoints[i] < 0)
        {
            w = w.reciprocal();
        }
        Goldilocks3GPU::mul(xi, *((Goldilocks3GPU::Element *)d_xiChallenge), w);

        if( k< NExtended){
            gl64_t * d_xDivXSub = (gl64_t *) (d_trace + xDivXSub_offset);
            Goldilocks3GPU::Element * xDivXSubComp = (Goldilocks3GPU::Element *) &d_xDivXSub[(k + i * NExtended) * FIELD_EXTENSION];
            Goldilocks3GPU::sub(*xDivXSubComp, d_x[k], xi);
            Goldilocks3GPU::inv( xDivXSubComp, xDivXSubComp);
            Goldilocks3GPU::mul(*xDivXSubComp, *xDivXSubComp, d_x[k]);
       }
    }


    
}

void calculateXDivXSub_inplace(uint64_t xDivXSub_offset, Goldilocks::Element *xiChallenge, SetupCtx &setupCtx, DeviceCommitBuffers* d_buffers){

    double time = omp_get_wtime();

    uint64_t nOpeningPoints = setupCtx.starkInfo.openingPoints.size();
    int64_t *openingPoints = setupCtx.starkInfo.openingPoints.data();
    gl64_t *x = (gl64_t *) setupCtx.proverHelpers.x;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    uint64_t nBits = setupCtx.starkInfo.starkStruct.nBits;


    gl64_t * d_xiChallenge;
    int64_t * d_openingPoints;
    cudaMalloc(&d_xiChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element));
    cudaMemcpy(d_xiChallenge, xiChallenge, FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    cudaMalloc(&d_openingPoints, nOpeningPoints * sizeof(int64_t));
    cudaMemcpy(d_openingPoints, openingPoints, nOpeningPoints * sizeof(int64_t), cudaMemcpyHostToDevice);
    gl64_t * d_x;
    cudaMalloc(&d_x, NExtended  * sizeof(Goldilocks::Element));
    cudaMemcpy(d_x, x, NExtended * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

    dim3 nThreads(32, 32);
    dim3 nBlocks((nOpeningPoints + nThreads.x - 1) / nThreads.x, (NExtended + nThreads.y - 1) / nThreads.y);
    calcXDivXSub<<<nBlocks, nThreads>>>(xDivXSub_offset, d_xiChallenge, Goldilocks::w(nBits).fe, nOpeningPoints, d_openingPoints, d_x, d_buffers->d_trace, NExtended);
}

void evmap_inplace(StepsParams& params, Goldilocks::Element *LEv){
}

__device__ void intt_tinny(gl64_t* data, uint32_t N, uint32_t logN, gl64_t* d_twiddles, uint32_t ncols)
{

    uint32_t halfN = N >> 1;
    // Reverse permutation
    for (uint32_t i = 0; i < N; i++)
    {
        uint32_t ibr = __brev(i) >> (32 - logN);
        if (ibr > i)
        {
            gl64_t tmp;
            for (uint32_t j = 0; j < ncols; j++)
            {
                tmp = data[i * ncols + j];
                data[i * ncols + j] = data[ibr * ncols + j];
                data[ibr * ncols + j] = tmp;
            }
        }
    }
    // Inverse NTT
    for (uint32_t i = 0; i < logN; i++)
    {
        for (uint32_t j = 0; j < halfN; j++)
        {
            for (uint32_t col = 0; col < ncols; col++)
            {
                uint32_t half_group_size = 1 << i;
                uint32_t group = j >> i;
                uint32_t offset = j & (half_group_size - 1);
                uint32_t index1 = (group << i + 1) + offset;
                uint32_t index2 = index1 + half_group_size;
                gl64_t factor = d_twiddles[offset * (N >> i + 1)];
                gl64_t odd_sub = gl64_t((uint64_t)data[index2 * ncols + col]) * factor;
                data[index2 * ncols + col] = gl64_t((uint64_t)data[index1 * ncols + col]) - odd_sub;
                data[index1 * ncols + col] = gl64_t((uint64_t)data[index1 * ncols + col]) + odd_sub;
            }
        }
    }
    // Scale by N^{-1}
    gl64_t factor = gl64_t(domain_size_inverse_[logN]);
    for (uint32_t i = 0; i < N * ncols; i++)
    {
        data[i] = gl64_t((uint64_t)data[i]) * factor;
    }
}

__global__ void fold(uint64_t step, gl64_t *friPol, gl64_t *d_challenge, gl64_t *d_ppar, gl64_t *d_twiddles, uint64_t shift_, uint64_t W_, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits){

    uint32_t polBits = prevBits;
    uint64_t sizePol = 1 << polBits;
    uint32_t foldedPolBits = currentBits;
    uint64_t sizeFoldedPol = 1 << foldedPolBits; 
    uint32_t ratio = sizePol/sizeFoldedPol;

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if( id < sizeFoldedPol){

        if (step == 0) return;
        gl64_t shift(shift_);
        gl64_t invShift = shift.reciprocal();
        for (uint32_t j = 0; j < nBitsExt - prevBits; j++)
        {
            invShift *= invShift;
        }
        
        gl64_t W(W_);
        gl64_t invW = W.reciprocal();
        // Evaluate the sinv value for the id current component
        gl64_t sinv = invShift;
        gl64_t base = invW;
        uint32_t exponent = id;

        while (exponent > 0) {
            if (exponent % 2 == 1) {
                sinv *= base;
            }
            base *= base;
            exponent /= 2;
        }

        gl64_t* ppar = (gl64_t *) d_ppar + id * ratio * FIELD_EXTENSION;
        for(int i = 0; i < ratio; i++){
            int ind = i * FIELD_EXTENSION;
            for(int k = 0; k < FIELD_EXTENSION; k++){
                ppar[ind + k].set_val(friPol[(i*sizeFoldedPol + id) * FIELD_EXTENSION + k]);
            }
        }
        intt_tinny(ppar, ratio, prevBits-currentBits, d_twiddles, FIELD_EXTENSION);
        
        // Multiply coefs by 1, shiftInv, shiftInv^2, shiftInv^3, ......
        gl64_t r(1);
        for (uint64_t i = 0; i < ratio; i++)
        {   
            Goldilocks3GPU::Element * component = (Goldilocks3GPU::Element *) &ppar[i * FIELD_EXTENSION];
            Goldilocks3GPU::mul(*component, *component, r);
            r *= sinv;
        }
        // evalPol
        if( ratio == 0){
            for(uint32_t i = 0; i < FIELD_EXTENSION; i++){
                friPol[id * FIELD_EXTENSION + i].set_val(0);
            }
        } else {
            for(uint32_t i = 0; i < FIELD_EXTENSION; i++){
                friPol[id * FIELD_EXTENSION + i] = ppar[(ratio -1)*FIELD_EXTENSION + i];
            }
            for(int i = ratio - 2; i >= 0; i--){
                Goldilocks3GPU::Element aux;
                Goldilocks3GPU::mul(aux, *((Goldilocks3GPU::Element *) &friPol[id * FIELD_EXTENSION]), *((Goldilocks3GPU::Element *) &d_challenge[0]));
                Goldilocks3GPU::add(*((Goldilocks3GPU::Element *) &friPol[id * FIELD_EXTENSION]), aux, *((Goldilocks3GPU::Element *) &ppar[i * FIELD_EXTENSION]));
            }
        }
    }

}

void fold_inplace(uint64_t step, uint64_t friPol_offset, Goldilocks::Element *challenge, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits, DeviceCommitBuffers* d_buffers){
    
    gl64_t * d_friPol = (gl64_t *) (d_buffers->d_trace + friPol_offset);
    gl64_t * d_challenge;
    gl64_t * d_ppar;
    gl64_t * d_twiddles;
    uint32_t ratio = 1 << (prevBits - currentBits);
    uint64_t halfRatio = ratio >> 1;


    uint64_t sizeFoldedPol = 1 << currentBits;

    CHECKCUDAERR(cudaMalloc(&d_challenge, FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_challenge, challenge, sizeof(Goldilocks::Element) * FIELD_EXTENSION, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMalloc(&d_ppar, (1 << prevBits) * FIELD_EXTENSION * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMalloc(&d_twiddles, halfRatio * sizeof(Goldilocks::Element)));

    //Generate inverse twiddle factors
    Goldilocks::Element* inv_twiddles= (Goldilocks::Element*) malloc(halfRatio * sizeof(Goldilocks::Element));
    Goldilocks::Element omega_inv = omegas_inv_[prevBits-currentBits];
    inv_twiddles[0] = Goldilocks::one();

    for (uint32_t i = 1; i < halfRatio; i++)
    {
        inv_twiddles[i] = inv_twiddles[i - 1] * omega_inv;
    }
    CHECKCUDAERR(cudaMemcpy(d_twiddles, inv_twiddles, halfRatio * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    free(inv_twiddles);


    dim3 nThreads(256);
    dim3 nBlocks((sizeFoldedPol) + nThreads.x - 1 / nThreads.x);
    fold<<<nBlocks, nThreads>>>(step, d_friPol, d_challenge, d_ppar, d_twiddles, Goldilocks::shift().fe,Goldilocks::w(prevBits).fe, nBitsExt, prevBits, currentBits);

    cudaFree(d_challenge);
    cudaFree(d_ppar);
    cudaFree(d_twiddles);
}

__global__ void transposeFRI(gl64_t* d_aux, gl64_t *pol, uint64_t degree, uint64_t width){
    uint64_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t height = degree / width;
    
    if(idx_x < width && idx_y < height){
        uint64_t fi = idx_y * width + idx_x;
        uint64_t di = idx_x * height + idx_y;
        for(uint64_t k = 0; k < FIELD_EXTENSION; k++){
            d_aux[di * FIELD_EXTENSION + k] = pol[fi * FIELD_EXTENSION + k];
        }
    }
}

void merkelizeFRI_inplace(uint64_t step, FRIProof<Goldilocks::Element> &proof, gl64_t * pol, MerkleTreeGL* treeFRI, uint64_t currentBits, uint64_t nextBits)
{
    uint64_t pol2N = 1 << currentBits;
    gl64_t *d_aux;
    cudaMalloc(&d_aux, pol2N * FIELD_EXTENSION * sizeof(Goldilocks::Element));

    uint64_t width = 1 << nextBits;
    uint64_t height = pol2N / width;
    dim3 nThreads(32, 32);
    dim3 nBlocks((width + nThreads.x - 1) / nThreads.x, (height + nThreads.y - 1) / nThreads.y);
    transposeFRI<<<nBlocks, nThreads>>>(d_aux, (gl64_t *) pol, pol2N, width);
    
    cudaMemcpy(treeFRI->source, d_aux, pol2N * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);

    uint64_t** d_tree = new uint64_t*[1];
    PoseidonGoldilocks::merkletree_cuda_gpudata_inplace(d_tree, (uint64_t *)d_aux, treeFRI->width, treeFRI->height);
    uint64_t tree_size = treeFRI->getNumNodes(treeFRI->height) * sizeof(uint64_t);
    CHECKCUDAERR(cudaMemcpy(treeFRI->get_nodes_ptr(), *d_tree, tree_size, cudaMemcpyDeviceToHost));
    treeFRI->getRoot(&proof.proof.fri.treesFRI[step].root[0]);

}

__global__ void getTreeTracePols(gl64_t* d_treeTrace, uint64_t traceWidth, uint64_t* d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth){

    uint64_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx_x < traceWidth && idx_y < nQueries){
        uint64_t row = d_friQueries[idx_y];
        uint64_t idx_trace = row * traceWidth + idx_x;
        uint64_t idx_buffer = idx_y * bufferWidth + idx_x;
        d_buffer[idx_buffer] = d_treeTrace[idx_trace];
    }

}  

void proveQueries_inplace(uint64_t* friQueries, uint64_t nQueries, FRIProof<Goldilocks::Element> &fproof, MerkleTreeGL **trees, uint64_t nTrees, DeviceCommitBuffers* d_buffers){

    uint64_t maxBuffSize = 0;
    for(uint64_t i = 0; i < nTrees; ++i) {
        uint64_t buffSize = trees[i]->getMerkleTreeWidth() + trees[i]->getMerkleProofSize();
        if(buffSize > maxBuffSize) {
            maxBuffSize = buffSize;
        }
    }
    Goldilocks::Element *buff = new Goldilocks::Element[maxBuffSize*nQueries*nTrees];
    gl64_t *d_buff;
    CHECKCUDAERR(cudaMalloc(&d_buff, maxBuffSize * nQueries * nTrees * sizeof(Goldilocks::Element)));
    uint64_t* d_friQueries;
    CHECKCUDAERR(cudaMalloc(&d_friQueries, nQueries * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(d_friQueries, friQueries, nQueries * sizeof(uint64_t), cudaMemcpyHostToDevice));

    int count = 0;
    for (uint k = 0; k < nTrees; k++){
        dim3 nThreads(32, 32);
        dim3 nBlocks((trees[k]->getMerkleTreeWidth() + nThreads.x - 1) / nThreads.x, (nQueries + nThreads.y - 1) / nThreads.y);
        if(k < nTrees - 1){
            getTreeTracePols<<<nBlocks, nThreads>>>(d_buffers->d_trace + trees[k]->souceTraceOffset, trees[k]->getMerkleTreeWidth(), d_friQueries, nQueries, d_buff + k * nQueries * maxBuffSize, maxBuffSize);
        } else {
            getTreeTracePols<<<nBlocks, nThreads>>>(&d_buffers->d_constTree[2], trees[k]->getMerkleTreeWidth(), d_friQueries, nQueries, d_buff + k * nQueries * maxBuffSize, maxBuffSize);

        }
    }
    CHECKCUDAERR(cudaMemcpy(buff, d_buff, maxBuffSize * nQueries * nTrees * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaFree(d_buff));
    CHECKCUDAERR(cudaFree(d_friQueries));

    count =0;
    for (uint k = 0; k < nTrees; k++){
        for (uint64_t i = 0; i < nQueries; i++)
        {
            trees[k]->genMerkleProof(&buff[count*maxBuffSize] + trees[k]->getMerkleTreeWidth(), friQueries[i], 0, trees[k]->getMerkleTreeHeight() * trees[k]->getMerkleTreeNFieldElements());
            ++count;

        }
    }
    count = 0;
    for (uint k = 0; k < nTrees; k++){
        for (uint64_t i = 0; i < nQueries; i++)
        {
            MerkleProof<Goldilocks::Element> mkProof(trees[k]->getMerkleTreeWidth(), trees[k]->getMerkleProofLength(), trees[k]->getNumSiblings(), &buff[count*maxBuffSize]);
            fproof.proof.fri.trees.polQueries[i].push_back(mkProof);
            ++count;
        }
    }

    delete[] buff;
    return;

}

template <typename ElementType>
void *genRecursiveProof_gpu(SetupCtx& setupCtx, json& globalInfo, uint64_t airgroupId, Goldilocks::Element *witness, Goldilocks::Element *pConstPols, Goldilocks::Element *pConstTree, Goldilocks::Element *publicInputs, std::string proofFile, DeviceCommitBuffers* d_buffers) { 

    TimerStart(STARK_PROOF);

    Goldilocks::Element *trace = new Goldilocks::Element[setupCtx.starkInfo.mapTotalN];
    CHECKCUDAERR(cudaMemset(d_buffers->d_trace, 0, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element)));

    FRIProof<Goldilocks::Element> proof(setupCtx.starkInfo);

    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;
    
    Starks<ElementType> starks(setupCtx, pConstTree);

    ExpressionsGPU expressionsCtx(setupCtx);

    uint64_t nFieldElements = setupCtx.starkInfo.starkStruct.verificationHashType == std::string("BN128") ? 1 : HASH_SIZE;

    TranscriptType transcript(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);

    Goldilocks::Element* evals = new Goldilocks::Element[setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* challenges = new Goldilocks::Element[setupCtx.starkInfo.challengesMap.size() * FIELD_EXTENSION];
    Goldilocks::Element* airgroupValues = nullptr;    

    StepsParams params = {
        trace: witness,
        pols : trace,
        publicInputs : publicInputs,
        challenges : challenges,
        airgroupValues : nullptr,
        evals : evals,
        xDivXSub : nullptr,
        pConstPolsAddress: pConstPols,
        pConstPolsExtendedTreeAddress: pConstTree,
    };

    StepsParams d_params = {
        trace: (Goldilocks::Element* ) d_buffers->d_witness,
        pols : (Goldilocks::Element* ) d_buffers->d_trace,
        publicInputs : (Goldilocks::Element* ) d_buffers->d_publicInputs,
        challenges : nullptr,
        airgroupValues : nullptr,
        evals : nullptr,
        xDivXSub : nullptr,
        pConstPolsAddress: (Goldilocks::Element* ) d_buffers->d_constPols,
        pConstPolsExtendedTreeAddress: (Goldilocks::Element* ) d_buffers->d_constTree,
    };

    //--------------------------------
    // 0.- Add const root and publics to transcript
    //--------------------------------
    TimerStart(STARK_STEP_0);
    ElementType verkey[nFieldElements];
    starks.treesGL[setupCtx.starkInfo.nStages + 1]->getRoot(verkey);
    starks.addTranscript(transcript, &verkey[0], nFieldElements);
    if(setupCtx.starkInfo.nPublics > 0) {
        if(!setupCtx.starkInfo.starkStruct.hashCommits) {
            starks.addTranscriptGL(transcript, &publicInputs[0], setupCtx.starkInfo.nPublics);
        } else {
            ElementType hash[nFieldElements];
            starks.calculateHash(hash, &publicInputs[0], setupCtx.starkInfo.nPublics);
            starks.addTranscript(transcript, hash, nFieldElements);
        }
    }
    TimerStopAndLog(STARK_STEP_0);

    TimerStart(STARK_STEP_1);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 1) {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    //Allocate d_tree
    uint64_t** d_tree = new uint64_t*[1];
    TimerStart(STARK_COMMIT_STAGE_1);
    starks.commitStage_inplace(1, d_buffers->d_witness, d_buffers->d_trace, d_tree, d_buffers);
    TimerStopAndLog(STARK_COMMIT_STAGE_1);

    /*Goldilocks::Element root[nFieldElements];
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    uint64_t tree_size = starks.treesGL[0]->getNumNodes(NExtended);
    CHECKCUDAERR(cudaMemcpy(&root[0],&d_tree[0][tree_size-nFieldElements], nFieldElements * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));*/
    
    offloadCommit(1, starks.treesGL, trace, d_buffers->d_trace, *d_tree, proof, setupCtx);
    starks.addTranscript(transcript, &proof.proof.roots[0][0], nFieldElements);
    
    TimerStopAndLog(STARK_STEP_1);

    TimerStart(STARK_STEP_2);
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++) {
        if(setupCtx.starkInfo.challengesMap[i].stage == 2) {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    //rick: tot aixo tambe ho puc posar sota del commit 1 no
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    Goldilocks::Element *num = new Goldilocks::Element[N*FIELD_EXTENSION];
    Goldilocks::Element *den = new Goldilocks::Element[N*FIELD_EXTENSION];
    Goldilocks::Element *gprod = new Goldilocks::Element[N*FIELD_EXTENSION];

    uint64_t gprodFieldId = setupCtx.expressionsBin.hints[0].fields[0].values[0].id;
    uint64_t numFieldId = setupCtx.expressionsBin.hints[0].fields[1].values[0].id;
    uint64_t denFieldId = setupCtx.expressionsBin.hints[0].fields[2].values[0].id;

    Dest numStruct(num, false);
    cudaMalloc(&numStruct.dest_gpu, N*FIELD_EXTENSION*sizeof(Goldilocks::Element));
    numStruct.addParams(setupCtx.expressionsBin.expressionsInfo[numFieldId]);
    
    Dest denStruct(den, false);
    cudaMalloc(&denStruct.dest_gpu, N*FIELD_EXTENSION*sizeof(Goldilocks::Element));
    denStruct.addParams(setupCtx.expressionsBin.expressionsInfo[denFieldId], true);
    std::vector<Dest> dests = {numStruct, denStruct};

    double time = omp_get_wtime();
    expressionsCtx.calculateExpressions_gpu(params, d_params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits));

    

    time = omp_get_wtime() - time;
    std::cout << "rick calculateExpressions time: " << time << std::endl;
    

    Goldilocks3::copy((Goldilocks3::Element *)&gprod[0], &Goldilocks3::one());
    for(uint64_t i = 1; i < N; ++i) {
        Goldilocks::Element res[3];
        Goldilocks3::mul((Goldilocks3::Element *)res, (Goldilocks3::Element *)&num[(i - 1) * FIELD_EXTENSION], (Goldilocks3::Element *)&den[(i - 1) * FIELD_EXTENSION]);
        Goldilocks3::mul((Goldilocks3::Element *)&gprod[i * FIELD_EXTENSION], (Goldilocks3::Element *)&gprod[(i - 1) * FIELD_EXTENSION], (Goldilocks3::Element *)res);
    }

    Goldilocks::Element *d_grod;
    cudaMalloc(&d_grod, N*FIELD_EXTENSION*sizeof(Goldilocks::Element));
    cudaMemcpy(d_grod, gprod, N*FIELD_EXTENSION*sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

    uint64_t offset = setupCtx.starkInfo.getTraceOffset("cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);
    uint64_t nCols = setupCtx.starkInfo.getTraceNColsSection("cm", setupCtx.starkInfo.cmPolsMap[gprodFieldId], false);

    dim3 nThreads(256);
    dim3 nBlocks((N + nThreads.x - 1) / nThreads.x);
    insertTracePol<<<nBlocks, nThreads>>>((Goldilocks::Element*) d_buffers->d_trace, offset, nCols, d_grod, FIELD_EXTENSION, N);
    
    delete num;
    delete den;
    delete gprod;

    time = omp_get_wtime();
    TimerStart(CALCULATE_IM_POLS);

    std::vector<Dest> dests2;
    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].imPol && setupCtx.starkInfo.cmPolsMap[i].stage == 2) {
            Goldilocks::Element* pols = setupCtx.starkInfo.cmPolsMap[i].stage == 1 ? params.trace : params.pols;
            uint64_t offset_ = setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(2), false)] + setupCtx.starkInfo.cmPolsMap[i].stagePos;
            Dest destStruct(&pols[offset_], setupCtx.starkInfo.mapSectionsN["cm" + to_string(2)]);
            destStruct.addParams(setupCtx.expressionsBin.expressionsInfo[setupCtx.starkInfo.cmPolsMap[i].expId], false);
            destStruct.dest_gpu = (Goldilocks::Element *)(d_buffers->d_trace + offset_);     
            dests2.push_back(destStruct);
        }
    }

    expressionsCtx.calculateExpressions_gpu2(params, d_params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests2, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits));

     TimerStopAndLog(CALCULATE_IM_POLS);
    time = omp_get_wtime() - time;
    std::cout << "rick calculateImPolsExpressions time: " << time << std::endl;
        
    TimerStart(STARK_COMMIT_STAGE_2);
    starks.commitStage_inplace(2, d_buffers->d_witness, d_buffers->d_trace, d_tree, d_buffers);
    offloadCommit(2, starks.treesGL, trace, d_buffers->d_trace, *d_tree, proof, setupCtx);

    TimerStopAndLog(STARK_COMMIT_STAGE_2);
    starks.addTranscript(transcript, &proof.proof.roots[1][0], nFieldElements);

    TimerStopAndLog(STARK_STEP_2);

    TimerStart(STARK_STEP_Q);

    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 1) {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }
    time = omp_get_wtime();

    uint64_t domainSize;
    uint64_t expressionId = setupCtx.starkInfo.cExpId;
    if(expressionId == setupCtx.starkInfo.cExpId || expressionId == setupCtx.starkInfo.friExpId) {
        setupCtx.expressionsBin.expressionsInfo[expressionId].destDim = 3;
        domainSize = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    } else {
        domainSize = 1 << setupCtx.starkInfo.starkStruct.nBits;
    }
    Dest destStructq(&params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]]);
    destStructq.addParams(setupCtx.expressionsBin.expressionsInfo[expressionId], false);
    destStructq.dest_gpu = (Goldilocks::Element *)(d_buffers->d_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("q", true)]);     
    std::vector<Dest> dests3 = {destStructq};
    expressionsCtx.calculateExpressions_gpu2(params, d_params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests3, domainSize);
    time = omp_get_wtime() - time;
    std::cout << "rick calculateExpression time: " << time << std::endl;

    TimerStart(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    time = omp_get_wtime();
    starks.commitStage_inplace(setupCtx.starkInfo.nStages + 1, nullptr, d_buffers->d_trace, d_tree, d_buffers);
    time = omp_get_wtime() - time;
    std::cout << "rick Q_inplace time: " << time << std::endl;
    time = omp_get_wtime();
    offloadCommit(setupCtx.starkInfo.nStages + 1, starks.treesGL, trace, d_buffers->d_trace, *d_tree, proof, setupCtx);
    time = omp_get_wtime() - time;
    std::cout << "rick Q_offloadCommit time: " << time << std::endl;
    TimerStopAndLog(STARK_COMMIT_QUOTIENT_POLYNOMIAL);
    starks.addTranscript(transcript, &proof.proof.roots[setupCtx.starkInfo.nStages][0], nFieldElements);
    TimerStopAndLog(STARK_STEP_Q);

    TimerStart(STARK_STEP_EVALS);

    uint64_t xiChallengeIndex = 0;
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 2) {
            if(setupCtx.starkInfo.challengesMap[i].stageId == 0) xiChallengeIndex = i;
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }

    Goldilocks::Element *xiChallenge = &challenges[xiChallengeIndex * FIELD_EXTENSION];
    Goldilocks::Element* LEv = &trace[setupCtx.starkInfo.mapOffsets[make_pair("LEv", true)]];

    computeLEv_inplace(xiChallenge, setupCtx.starkInfo.mapOffsets[make_pair("LEv", true)], setupCtx.starkInfo.starkStruct.nBits, setupCtx.starkInfo.openingPoints.size(), setupCtx.starkInfo.openingPoints.data(), d_buffers);

    ////////// copy trace downwards
    time = omp_get_wtime();
    CHECKCUDAERR(cudaMemcpy(trace, d_buffers->d_trace, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    time = omp_get_wtime() - time;
    std::cout << "rick copy trace time: " << time << std::endl;
    //////////

    starks.computeEvals(params ,LEv, proof);
    //evmap_inplace(params, LEv); // rick: pendent
    //computeEvals_inplace(params, LEv, proof, d_buffers);

    time = omp_get_wtime();
    if(!setupCtx.starkInfo.starkStruct.hashCommits) {
        starks.addTranscriptGL(transcript, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
    } else {
        ElementType hash[nFieldElements];
        starks.calculateHash(hash, evals, setupCtx.starkInfo.evMap.size() * FIELD_EXTENSION);
        starks.addTranscript(transcript, hash, nFieldElements);
    }

    // Challenges for FRI polynomial
    for (uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); i++)
    {
        if(setupCtx.starkInfo.challengesMap[i].stage == setupCtx.starkInfo.nStages + 3) {
            starks.getChallenge(transcript, challenges[i * FIELD_EXTENSION]);
        }
    }
    time = omp_get_wtime() - time;
    std::cout << "rick evals-challenges time: " << time << std::endl;

    TimerStopAndLog(STARK_STEP_EVALS);

    //--------------------------------
    // 6. Compute FRI
    //--------------------------------
    TimerStart(STARK_STEP_FRI);

    ////////// copy trace upwards
    time = omp_get_wtime();
    CHECKCUDAERR(cudaMemcpy(d_buffers->d_trace, trace, setupCtx.starkInfo.mapTotalN * sizeof(Goldilocks::Element), cudaMemcpyHostToDevice));
    time = omp_get_wtime() - time;
    std::cout << "rick copy trace time: " << time << std::endl;
    //////////

    TimerStart(COMPUTE_FRI_POLYNOMIAL);
    uint64_t xDivXSub_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("xDivXSubXi", true)];
    params.xDivXSub = &trace[xDivXSub_offset];
    d_params.xDivXSub = (Goldilocks::Element *)(d_buffers->d_trace + xDivXSub_offset);
    time = omp_get_wtime();
    calculateXDivXSub_inplace(xDivXSub_offset, xiChallenge, setupCtx, d_buffers);
    time = omp_get_wtime() - time;
    std::cout << "rick calculateXDivXSub time: " << time << std::endl;

    // FRI exoressuibs
    expressionId = setupCtx.starkInfo.friExpId;
    if(expressionId == setupCtx.starkInfo.cExpId || expressionId == setupCtx.starkInfo.friExpId) {
        setupCtx.expressionsBin.expressionsInfo[expressionId].destDim = 3;
        domainSize = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
    } else {
        domainSize = 1 << setupCtx.starkInfo.starkStruct.nBits;
    }
    Dest destStructf(&params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]]);
    destStructf.addParams(setupCtx.expressionsBin.expressionsInfo[expressionId], false);
    destStructf.dest_gpu = (Goldilocks::Element *)(d_buffers->d_trace + setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)]);
    std::vector<Dest> destsf = {destStructf};
    expressionsCtx.calculateExpressions_gpu2(params, d_params, setupCtx.expressionsBin.expressionsBinArgsExpressions, destsf, domainSize);


    TimerStopAndLog(COMPUTE_FRI_POLYNOMIAL);

    Goldilocks::Element challenge[FIELD_EXTENSION];
    uint64_t friPol_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("f", true)];
    Goldilocks::Element *friPol = &trace[friPol_offset];
    gl64_t *d_friPol = (gl64_t *) (d_buffers->d_trace + friPol_offset);
    
    CHECKCUDAERR(cudaMemcpy(friPol, d_buffers->d_trace + friPol_offset, NExtended * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
        
    TimerStart(STARK_FRI_FOLDING);
    uint64_t nBitsExt =  setupCtx.starkInfo.starkStruct.steps[0].nBits;
    for (uint64_t step = 0; step < setupCtx.starkInfo.starkStruct.steps.size(); step++)
    {   
        uint64_t currentBits = setupCtx.starkInfo.starkStruct.steps[step].nBits;
        uint64_t prevBits = step == 0 ? currentBits : setupCtx.starkInfo.starkStruct.steps[step - 1].nBits;
        fold_inplace(step, friPol_offset, challenge, nBitsExt, prevBits, currentBits, d_buffers);

        if (step < setupCtx.starkInfo.starkStruct.steps.size() - 1)
        {
            merkelizeFRI_inplace(step, proof, d_friPol, starks.treesFRI[step], currentBits, setupCtx.starkInfo.starkStruct.steps[step + 1].nBits);
            starks.addTranscript(transcript, &proof.proof.fri.treesFRI[step].root[0], nFieldElements);
        }
        else
        {
            CHECKCUDAERR(cudaMemcpy(friPol, d_friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
            if(!setupCtx.starkInfo.starkStruct.hashCommits) {
                starks.addTranscriptGL(transcript, friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
            } else {
                ElementType hash[nFieldElements];
                starks.calculateHash(hash, friPol, (1 << setupCtx.starkInfo.starkStruct.steps[step].nBits) * FIELD_EXTENSION);
                starks.addTranscript(transcript, hash, nFieldElements);
            } 
            
        }
        starks.getChallenge(transcript, *challenge);
    }
    TimerStopAndLog(STARK_FRI_FOLDING);

    TimerStart(STARK_FRI_QUERIES);
    uint64_t friQueries[setupCtx.starkInfo.starkStruct.nQueries];
    TranscriptType transcriptPermutation(setupCtx.starkInfo.starkStruct.merkleTreeArity, setupCtx.starkInfo.starkStruct.merkleTreeCustom);
    starks.addTranscriptGL(transcriptPermutation, challenge, FIELD_EXTENSION);
    transcriptPermutation.getPermutations(friQueries, setupCtx.starkInfo.starkStruct.nQueries, setupCtx.starkInfo.starkStruct.steps[0].nBits);
    time = omp_get_wtime() - time;
    
    uint64_t nTrees = setupCtx.starkInfo.nStages + setupCtx.starkInfo.customCommits.size() + 2;
    proveQueries_inplace(friQueries, setupCtx.starkInfo.starkStruct.nQueries, proof, starks.treesGL, nTrees, d_buffers);

    CHECKCUDAERR(cudaMemcpy(friPol, d_buffers->d_trace + friPol_offset, NExtended * FIELD_EXTENSION * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
   
    for(uint64_t step = 1; step < setupCtx.starkInfo.starkStruct.steps.size(); ++step) {
        FRI<Goldilocks::Element>::proveFRIQueries(friQueries, setupCtx.starkInfo.starkStruct.nQueries, step, setupCtx.starkInfo.starkStruct.steps[step].nBits, proof, starks.treesFRI[step - 1]);
    }

    FRI<ElementType>::setFinalPol(proof, friPol, setupCtx.starkInfo.starkStruct.steps[setupCtx.starkInfo.starkStruct.steps.size() - 1].nBits);
    TimerStopAndLog(STARK_FRI_QUERIES);

    TimerStopAndLog(STARK_STEP_FRI);
    
    delete challenges;
    delete evals;
    delete airgroupValues;
    
    time = omp_get_wtime();
    nlohmann::json jProof = proof.proof.proof2json();
    nlohmann::json zkin = proof2zkinStark(jProof, setupCtx.starkInfo);

    if(!proofFile.empty()) {
        json2file(jProof, proofFile);
    }
    time = omp_get_wtime() - time;
    std::cout << "rick proof2json time: " << time << std::endl;

    TimerStopAndLog(STARK_PROOF);

    zkin = publics2zkin(zkin, publicInputs, globalInfo, airgroupId);

    delete trace;
    return (void *) new nlohmann::json(zkin);
}
#endif
