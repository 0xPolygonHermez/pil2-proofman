#ifndef STARKS_GPU_HPP
#define STARKS_GPU_HPP

#include "expressions_gpu.cuh"
class gl64_t;

extern Goldilocks::Element omegas_inv_[33];

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


struct EvalInfo
{
    uint64_t type; // 0: cm, 1: custom, 2: fixed
    uint64_t offset;
    uint64_t stride;
    uint64_t dim;
    uint64_t openingPos;
    uint64_t evalPos;
};


__global__ void insertTracePol(Goldilocks::Element *d_aux_trace, uint64_t offset, uint64_t stride, Goldilocks::Element *d_pol, uint64_t dim, uint64_t N);

__global__ void fillLEv_2d(gl64_t* d_LEv, gl64_t *d_xiChallenge, uint64_t W_, uint64_t nOpeningPoints, int64_t *d_openingPoints, uint64_t shift_, uint64_t N);

void computeLEv_inplace(Goldilocks::Element *xiChallenge, uint64_t nBits, uint64_t nOpeningPoints, int64_t *openingPoints, DeviceCommitBuffers *d_buffers, uint64_t offset_helper, gl64_t* d_LEv, double *nttTime=nullptr);

void calculateXis_inplace(SetupCtx &setupCtx, StepsParams &h_params, Goldilocks::Element *xiChallenge);

__global__ void computeEvals_v2(
    uint64_t extendBits,
    uint64_t size_eval,
    uint64_t N,
    uint64_t openingsSize,
    uint64_t LEv_offset,
    gl64_t *d_evals,
    EvalInfo *d_evalInfo,
    gl64_t *d_cmPols,
    gl64_t *d_customComits,
    gl64_t *d_fixedPols);

void evmap_inplace(Goldilocks::Element * evals, StepsParams &d_params, FRIProof<Goldilocks::Element> &proof, Starks<Goldilocks::Element> *starks, DeviceCommitBuffers *d_buffers, uint64_t nOpeningPoints, int64_t *openingPoints, Goldilocks::Element *d_LEv_);

__device__ void intt_tinny(gl64_t *data, uint32_t N, uint32_t logN, gl64_t *d_twiddles, uint32_t ncols);

__global__ void fold(uint64_t step, gl64_t *friPol, gl64_t *d_challenge, gl64_t *d_ppar, gl64_t *d_twiddles, uint64_t shift_, uint64_t W_, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits);

void fold_inplace(uint64_t step, uint64_t friPol_offset, uint64_t offset_helper, Goldilocks::Element *challenge, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits, DeviceCommitBuffers *d_buffers);

__global__ void transposeFRI(gl64_t *d_aux, gl64_t *pol, uint64_t degree, uint64_t width);

void merkelizeFRI_inplace(SetupCtx& setupCtx, StepsParams &d_param, uint64_t step, FRIProof<Goldilocks::Element> &proof, gl64_t *pol, MerkleTreeGL *treeFRI, uint64_t currentBits, uint64_t nextBits, double * merkleTime=nullptr);

__global__ void getTreeTracePols(gl64_t *d_treeTrace, uint64_t traceWidth, uint64_t *d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth);

__device__ void genMerkleProof_(gl64_t *nodes, gl64_t *proof, uint64_t idx, uint64_t offset, uint64_t n, uint64_t nFieldElements);

__global__ void genMerkleProof(gl64_t *d_nodes, uint64_t sizeLeaves, uint64_t *d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth, uint64_t maxTreeWidth, uint64_t nFieldElements);

void proveQueries_inplace(SetupCtx& setupCtx, uint64_t *friQueries, uint64_t nQueries, FRIProof<Goldilocks::Element> &fproof, MerkleTreeGL **trees, uint64_t nTrees, DeviceCommitBuffers *d_buffers, gl64_t* d_const_tree, uint32_t nStages, StepsParams &d_params);
void proveFRIQueries_inplace(SetupCtx& setupCtx, uint64_t step, uint64_t currentBits, uint64_t *friQueries, uint64_t nQueries, FRIProof<Goldilocks::Element> &fproof, MerkleTreeGL *treeFRI);

void calculateImPolsExpressions(SetupCtx& setupCtx, ExpressionsGPU& expressionsCtx, StepsParams& h_params, StepsParams *d_params, int64_t step);

void calculateExpression(SetupCtx& setupCtx, ExpressionsGPU& expressionsCtx, StepsParams* d_params,Goldilocks::Element* dest_gpu, uint64_t expressionId, bool inverse = false);

void calculateFRIExpression(SetupCtx& setupCtx, StepsParams* d_params);
void offloadCommit(uint64_t step, MerkleTreeGL **treesGL, gl64_t *d_aux_trace, FRIProof<Goldilocks::Element> &proof, SetupCtx &setupCtx);
void offloadCommitFRI(uint64_t step, MerkleTreeGL *treeFRI, FRIProof<Goldilocks::Element> &proof, SetupCtx &setupCtx);
#endif