#ifndef STARKS_GPU_HPP
#define STARKS_GPU_HPP

#include "expressions_gpu.cuh"
#include "transcriptGL.cuh"
#include "ntt_goldilocks.cuh"
#include "poseidon2_goldilocks.cuh"
#include "hints.cuh"

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

__global__ void setProdIdentity3(gl64_t *pol);

__global__ void computeX_kernel(gl64_t *x, uint64_t NExtended, Goldilocks::Element shift, Goldilocks::Element w);

__global__ void insertTracePol(Goldilocks::Element *d_aux_trace, uint64_t offset, uint64_t stride, Goldilocks::Element *d_pol, uint64_t dim, uint64_t N);

__global__ void fillLEv_2d(gl64_t* d_LEv, gl64_t *d_xiChallenge, uint64_t W_, uint64_t nOpeningPoints, int64_t *d_openingPoints, uint64_t shift_, uint64_t N);

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

__device__ void intt_tinny(gl64_t *data, uint32_t N, uint32_t logN, gl64_t *d_twiddles, uint32_t ncols);

__global__ void fold(uint64_t step, gl64_t *friPol, gl64_t *d_challenge, gl64_t *d_ppar, Goldilocks::Element omega_inv, gl64_t *d_twiddles, uint64_t shift_, uint64_t W_, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits);

__global__ void transposeFRI(gl64_t *d_aux, gl64_t *pol, uint64_t degree, uint64_t width);

__global__ void getTreeTracePols(gl64_t *d_treeTrace, uint64_t traceWidth, uint64_t *d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth);

__device__ void genMerkleProof_(gl64_t *nodes, gl64_t *proof, uint64_t idx, uint64_t offset, uint64_t n, uint64_t nFieldElements);

__global__ void genMerkleProof(gl64_t *d_nodes, uint64_t sizeLeaves, uint64_t *d_friQueries, uint64_t nQueries, gl64_t *d_buffer, uint64_t bufferWidth, uint64_t maxTreeWidth, uint64_t nFieldElements);

__global__ void computeX_kernel(gl64_t *x, uint64_t NExtended, Goldilocks::Element shift, Goldilocks::Element w);
__global__ void buildZHInv_kernel(gl64_t *d_zi, uint64_t extend, uint64_t NExtended, Goldilocks::Element sn, Goldilocks::Element w);

__global__ void moduleQueries(uint64_t* d_friQueries, uint64_t nQueries, uint64_t currentBits);

void unpack_trace(PackedInfo *packedInfo, uint64_t* src, uint64_t* dst, uint64_t nCols, uint64_t nRows, cudaStream_t stream, TimerGPU &timer);
void computeLEv_inplace(Goldilocks::Element *d_xiChallenge, uint64_t nBits, uint64_t nOpeningPoints, int64_t *d_openingPoints, gl64_t *d_aux_trace, uint64_t offset_helper, gl64_t* d_LEv, TimerGPU &timer, cudaStream_t stream);

void calculateXis_inplace(SetupCtx &setupCtx, StepsParams &h_params, int64_t *d_openingPoints, Goldilocks::Element *d_xiChallenge, cudaStream_t stream);

void commitStage_inplace(uint64_t step, SetupCtx& setupCtx, MerkleTreeGL**treesGL, gl64_t *d_witness, gl64_t *d_aux_trace, TranscriptGL_GPU *d_transcript, bool skipRecalculation, TimerGPU &timer, cudaStream_t stream);
void extendAndMerkelize_inplace(uint64_t step, SetupCtx& setupCtx, MerkleTreeGL **treesGL, gl64_t *d_witness, gl64_t *d_aux_trace, TranscriptGL_GPU *d_transcript, bool skipRecalculation, TimerGPU &timer, cudaStream_t stream);
void extendAndMerkelizeFixed(SetupCtx&, Goldilocks::Element *d_fixedPols, Goldilocks::Element *d_fixedPolsExtended, TimerGPU &timer, cudaStream_t stream);
void computeQ_inplace(uint64_t step, SetupCtx& setupCtx, MerkleTreeGL **treesGL, gl64_t *d_aux_trace, TranscriptGL_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream);

void computeZerofier(Goldilocks::Element *d_zi, uint64_t nBits, uint64_t nBitsExt, cudaStream_t stream);

void evmap_inplace(SetupCtx &setupCtx, StepsParams &h_params, uint64_t chunk, uint64_t nOpeningPoints, int64_t *openingPoints, AirInstanceInfo *air_instance_info, Goldilocks::Element *d_LEv, uint64_t offset_helper, TimerGPU &timer, cudaStream_t stream);

void fold_inplace(uint64_t step, uint64_t friPol_offset, uint64_t offset_helper, Goldilocks::Element *challenge, uint64_t nBitsExt, uint64_t prevBits, uint64_t currentBits, gl64_t *d_aux_trace, TimerGPU &timer, cudaStream_t stream);

void merkelizeFRI_inplace(SetupCtx& setupCtx, StepsParams &d_param, uint64_t step, gl64_t *pol, MerkleTreeGL *treeFRI, uint64_t currentBits, uint64_t nextBits, TranscriptGL_GPU *d_transcript, TimerGPU &timer, cudaStream_t stream);

void proveQueries_inplace(SetupCtx& setupCtx, gl64_t *d_queries_buff, uint64_t *friQueries, uint64_t nQueries, MerkleTreeGL **trees, uint64_t nTrees, gl64_t *d_aux_trace, gl64_t* d_const_tree, uint32_t nStages, cudaStream_t stream);
void proveFRIQueries_inplace(SetupCtx& setupCtx, gl64_t *d_queries_buff, uint64_t step, uint64_t currentBits, uint64_t *friQueries, uint64_t nQueries, MerkleTreeGL *treeFRI, cudaStream_t stream);

void calculateImPolsExpressions(SetupCtx& setupCtx, ExpressionsGPU *expressionsCtx, StepsParams &h_params, StepsParams *d_params, int64_t step, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU& timer, cudaStream_t stream);

void calculateExpression(SetupCtx& setupCtx, ExpressionsGPU* expressionsCtx, StepsParams* d_params, Goldilocks::Element* dest_gpu, uint64_t expressionId, bool inverse, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU& timer, cudaStream_t stream, bool debug = false);

void calculateFRIExpression(SetupCtx& setupCtx, StepsParams &h_params, AirInstanceInfo *air_instance_info, cudaStream_t stream);

void calculateHash(TranscriptGL_GPU *d_transcript, Goldilocks::Element* hash, SetupCtx &setupCtx, Goldilocks::Element* buffer, uint64_t nElements, cudaStream_t stream);

void setProof(SetupCtx &setupCtx, Goldilocks::Element *h_aux_trace, Goldilocks::Element *proof_buffer_pinned, cudaStream_t stream);
void writeProof(SetupCtx &setupCtx, Goldilocks::Element *proof_buffer_pinned, uint64_t *proof_buffer, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, std::string proofFile);
#endif