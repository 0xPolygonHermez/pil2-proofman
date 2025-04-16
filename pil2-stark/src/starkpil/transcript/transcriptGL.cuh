
#ifndef TRANSCRIPT_GPU_CLASS
#define TRANSCRIPT_GPU_CLASS

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "poseidon2_goldilocks.hpp"
#include "zklog.hpp"

#define TRANSCRIPT_STATE_SIZE 4
#define TRANSCRIPT_PENDING_SIZE 8
#define TRANSCRIPT_OUT_SIZE 12

__device__ void _updateState(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor);
__global__ void _add(Goldilocks::Element* input, uint64_t size, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor);

class TranscriptGL_GPU
{
    

public:
    Goldilocks::Element* state;
    Goldilocks::Element* pending;
    Goldilocks::Element* out;

    uint *pending_cursor;
    uint *out_cursor;
    uint *state_cursor;

    TranscriptGL_GPU(uint64_t arity, bool custom)
    {
        cudaMalloc((void**)&state, TRANSCRIPT_OUT_SIZE * sizeof(Goldilocks::Element));
        cudaMalloc((void**)&pending, TRANSCRIPT_PENDING_SIZE * sizeof(Goldilocks::Element));
        cudaMalloc((void**)&out, TRANSCRIPT_OUT_SIZE * sizeof(Goldilocks::Element));
        cudaMalloc((void**)&pending_cursor, sizeof(uint));
        cudaMalloc((void**)&out_cursor, sizeof(uint));
        cudaMalloc((void**)&state_cursor, sizeof(uint));

        cudaMemset(state, 0, TRANSCRIPT_OUT_SIZE * sizeof(Goldilocks::Element));
        cudaMemset(pending, 0, TRANSCRIPT_PENDING_SIZE * sizeof(Goldilocks::Element));
        cudaMemset(out, 0, TRANSCRIPT_OUT_SIZE * sizeof(Goldilocks::Element));
        cudaMemset(pending_cursor, 0, sizeof(uint));
        cudaMemset(out_cursor, 0, sizeof(uint));
        cudaMemset(state_cursor, 0, sizeof(uint));
    }
    ~TranscriptGL_GPU()
    {
        cudaFree(state);
        cudaFree(pending);
        cudaFree(out);
        cudaFree(pending_cursor);
        cudaFree(out_cursor);
        cudaFree(state_cursor);
    }
    void put(Goldilocks::Element *input, uint64_t size);
    void getField(uint64_t *output);
    void getState(Goldilocks::Element* output);
    void getState(Goldilocks::Element* output, uint64_t nOutputs);
    void getPermutations(uint64_t *res, uint64_t n, uint64_t nBits);
    Goldilocks::Element getFields1();
};

#endif