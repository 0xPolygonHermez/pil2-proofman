#ifndef TRANSCRIPT_BN128_GPU_CLASS
#define TRANSCRIPT_BN128_GPU_CLASS

#include "bn128.cuh"
#include "goldilocks_base_field.hpp"
#include "poseidon/poseidon_bn128.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "zklog.hpp"
#include "gpu_timer.cuh"



// Device functions
__device__ void _updateStateBN128(PoseidonBN128GPU::FrElement* state, PoseidonBN128GPU::FrElement* pending, PoseidonBN128GPU::FrElement* out, uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity);
__device__ PoseidonBN128GPU::FrElement _getFields253(PoseidonBN128GPU::FrElement* state, PoseidonBN128GPU::FrElement* pending, PoseidonBN128GPU::FrElement* out, uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity);
__device__ uint64_t _getFields1BN128(PoseidonBN128GPU::FrElement* state, PoseidonBN128GPU::FrElement* pending, PoseidonBN128GPU::FrElement* out, uint64_t* out3, uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity);

// Global kernels
__global__ void _addBN128_GL(Goldilocks::Element* input, uint64_t size, PoseidonBN128GPU::FrElement* state, PoseidonBN128GPU::FrElement* pending, PoseidonBN128GPU::FrElement* out, uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity);
__global__ void _addBN128_Fr(PoseidonBN128GPU::FrElement* input, uint64_t size, PoseidonBN128GPU::FrElement* state, PoseidonBN128GPU::FrElement* pending, PoseidonBN128GPU::FrElement* out, uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity);
__global__ void _getFieldBN128(uint64_t* output, PoseidonBN128GPU::FrElement* state, PoseidonBN128GPU::FrElement* pending, PoseidonBN128GPU::FrElement* out, uint64_t* out3, uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity);
__global__ void __getStateBN128(PoseidonBN128GPU::FrElement* output, PoseidonBN128GPU::FrElement* state, PoseidonBN128GPU::FrElement* pending, PoseidonBN128GPU::FrElement* out, uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity);
__global__ void __getPermutationsBN128(uint64_t *res, uint64_t n, uint64_t nBits, PoseidonBN128GPU::FrElement* state, PoseidonBN128GPU::FrElement* pending, PoseidonBN128GPU::FrElement* out, uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity);

class TranscriptBN128_GPU
{
public:
    uint64_t arity;

    // GPU device memory pointers
    PoseidonBN128GPU::FrElement* state;
    PoseidonBN128GPU::FrElement* pending;
    PoseidonBN128GPU::FrElement* out;
    uint64_t* out3;

    uint *pending_cursor;
    uint *out_cursor;
    uint *out3_cursor;

    TranscriptBN128_GPU(uint64_t arity, bool custom, cudaStream_t stream);
    ~TranscriptBN128_GPU()
    {
        CHECKCUDAERR(cudaFree(state));
        CHECKCUDAERR(cudaFree(pending));
        CHECKCUDAERR(cudaFree(out));
        CHECKCUDAERR(cudaFree(out3));
        CHECKCUDAERR(cudaFree(pending_cursor));
        CHECKCUDAERR(cudaFree(out_cursor));
        CHECKCUDAERR(cudaFree(out3_cursor));
    }
    
    void reset(cudaStream_t stream);

    void put(Goldilocks::Element *input, uint64_t size, cudaStream_t stream, TimerGPU *timer=nullptr);
    void put(PoseidonBN128GPU::FrElement *input, uint64_t size, cudaStream_t stream, TimerGPU *timer=nullptr);
    void getField(uint64_t *output, cudaStream_t stream);
    void getState(PoseidonBN128GPU::FrElement* output, cudaStream_t stream);
    void getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, cudaStream_t stream);
    
    // Initializes Poseidon GPU constants
    static void init_const(uint32_t* gpu_ids, uint32_t num_gpu_ids, uint32_t arity);
};

#endif
