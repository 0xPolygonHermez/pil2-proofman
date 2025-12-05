
#ifndef TRANSCRIPT_GPU_CLASS
#define TRANSCRIPT_GPU_CLASS

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "poseidon2_goldilocks.hpp"
#include "zklog.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "transcriptGL.hpp"

__device__ void _updateState(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor);
__device__ Goldilocks::Element _getFields1(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor);__global__ void _add(Goldilocks::Element* input, uint64_t size, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor);
__global__ void _getField(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor);
__global__ void __getState(Goldilocks::Element* output, uint64_t nOutputs, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor);
__global__ void __getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor);

class TranscriptGL_GPU
{
    

public:
    Goldilocks::Element* state;
    Goldilocks::Element* pending;
    Goldilocks::Element* out;

    uint *pending_cursor;
    uint *out_cursor;
    uint *state_cursor;

    TranscriptGL_GPU(uint64_t arity, bool custom, cudaStream_t stream);
    ~TranscriptGL_GPU()
    {
        CHECKCUDAERR(cudaFree(state));
        CHECKCUDAERR(cudaFree(pending));
        CHECKCUDAERR(cudaFree(out));
        CHECKCUDAERR(cudaFree(pending_cursor));
        CHECKCUDAERR(cudaFree(out_cursor));
        CHECKCUDAERR(cudaFree(state_cursor));
    }
    
    void reset(cudaStream_t stream) {
        cudaMemsetAsync(state, 0, TRANSCRIPT_OUT_SIZE * sizeof(Goldilocks::Element), stream);
        cudaMemsetAsync(pending, 0, TRANSCRIPT_PENDING_SIZE * sizeof(Goldilocks::Element), stream);
        cudaMemsetAsync(out, 0, TRANSCRIPT_OUT_SIZE * sizeof(Goldilocks::Element), stream);
        cudaMemsetAsync(pending_cursor, 0, sizeof(uint), stream);
        cudaMemsetAsync(out_cursor, 0, sizeof(uint), stream);
        cudaMemsetAsync(state_cursor, 0, sizeof(uint), stream);
    };

    void put(Goldilocks::Element *input, uint64_t size, cudaStream_t stream);
    void getField(uint64_t *output, cudaStream_t stream);
    void getState(Goldilocks::Element* output, cudaStream_t stream);
    void getState(Goldilocks::Element* output, uint64_t nOutputs, cudaStream_t stream);
    void getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, cudaStream_t stream);
    static void init_const(uint32_t* gpu_ids, uint32_t num_gpu_ids);
    
};

#endif