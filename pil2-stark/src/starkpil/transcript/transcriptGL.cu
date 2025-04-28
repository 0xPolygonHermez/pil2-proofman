#include "transcriptGL.cuh"
#include "poseidon2_goldilocks.cuh"

#include "math.h"

__device__ __constant__ gl64_t GPU_C[118];
__device__ __constant__ gl64_t GPU_D[12];

void TranscriptGL_GPU::put(Goldilocks::Element *input, uint64_t size)
{
   _add<<<1,1>>>(input, size, state, pending, out, pending_cursor, out_cursor, state_cursor);
}

__device__ void _updateState(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor) 
{

    static int initialized = 0;

    if (initialized == 0)
    {
        initialized = 1;
        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C, Poseidon2GoldilocksConstants::C, 118 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D, Poseidon2GoldilocksConstants::D, 12 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
    
    }
    
    while(*pending_cursor < TRANSCRIPT_PENDING_SIZE) {
        pending[*pending_cursor].fe = 0;
        *pending_cursor++;
    }
    Goldilocks::Element inputs[TRANSCRIPT_OUT_SIZE];
    for (int i = 0; i < TRANSCRIPT_PENDING_SIZE; i++)
    {
        inputs[i] = pending[i];
    }
    for (int i = 0; i < TRANSCRIPT_STATE_SIZE; i++)
    {
        inputs[i + TRANSCRIPT_PENDING_SIZE] = state[i];
    }
    hash_full_result_seq_2(out, inputs, GPU_C, GPU_D);

    *out_cursor = TRANSCRIPT_OUT_SIZE;
    for (int i = 0; i < TRANSCRIPT_PENDING_SIZE; i++)
    {
        pending[i].fe = 0;
    }
    *pending_cursor = 0;
    for (int i = 0; i < TRANSCRIPT_OUT_SIZE; i++)
    {
        state[i] = out[i];
    }
}


__global__ void _add(Goldilocks::Element* input, uint64_t size,  Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor)
{
    for (uint64_t i = 0; i < size; i++)
    {
        pending[*pending_cursor] = input[i];
        *pending_cursor++;
        *out_cursor = 0;
        if (*pending_cursor == TRANSCRIPT_PENDING_SIZE)
        {
            _updateState(state, pending, out, pending_cursor, out_cursor, state_cursor);
        }
    }
}

void TranscriptGL_GPU::getField(uint64_t* output)
{
    for (int i = 0; i < 3; i++)
    {
        Goldilocks::Element val = getFields1();
        output[i] = val.fe;
    }
}

void TranscriptGL_GPU::getState(Goldilocks::Element* output) {
    /*if(pending_cursor > 0) {
        _updateState();
    }
    std::memcpy(output, state, TRANSCRIPT_STATE_SIZE * sizeof(Goldilocks::Element));*/
}

void TranscriptGL_GPU::getState(Goldilocks::Element* output, uint64_t nOutputs) {
    /*if(pending_cursor > 0) {
        _updateState();
    }
    std::memcpy(output, state, nOutputs * sizeof(Goldilocks::Element));8*/
}

Goldilocks::Element TranscriptGL_GPU::getFields1()
{
    /*if (out_cursor == 0)
    {
        _updateState();
    }
    Goldilocks::Element res = out[(TRANSCRIPT_OUT_SIZE - out_cursor) % TRANSCRIPT_OUT_SIZE];
    out_cursor--;
    return res;*/
}

void TranscriptGL_GPU::getPermutations(uint64_t *res, uint64_t n, uint64_t nBits)
{
    /*uint64_t totalBits = n * nBits;

    uint64_t NFields = floor((float)(totalBits - 1) / 63) + 1;
    Goldilocks::Element fields[NFields];

    for (uint64_t i = 0; i < NFields; i++)
    {
        fields[i] = getFields1();
    }
    
    std::string permutation = " ";

    uint64_t curField = 0;
    uint64_t curBit = 0;
    for (uint64_t i = 0; i < n; i++)
    {
        uint64_t a = 0;
        for (uint64_t j = 0; j < nBits; j++)
        {
            uint64_t bit = (Goldilocks::toU64(fields[curField]) >> curBit) & 1;
            if (bit)
                a = a + (1 << j);
            curBit++;
            if (curBit == 63)
            {
                curBit = 0;
                curField++;
            }
        }
        res[i] = a;
        permutation += std::to_string(a) + " ";
    }*/
}
