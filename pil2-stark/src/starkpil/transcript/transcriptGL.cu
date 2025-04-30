#include "transcriptGL.cuh"
#include "poseidon2_goldilocks.cuh"

#include "math.h"

__device__ __constant__ gl64_t GPU_C[118];
__device__ __constant__ gl64_t GPU_D[12];


__device__ void _updateState(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor) 
{
    
    while(*pending_cursor < TRANSCRIPT_PENDING_SIZE) {
        pending[*pending_cursor].fe = 0;
        (*pending_cursor) += 1;
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
    hash_full_result_seq_2((gl64_t*)out, (gl64_t*)inputs, GPU_C, GPU_D);

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

__device__ Goldilocks::Element _getFields1(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor){
    if (*out_cursor == 0)
    {
        _updateState(state, pending, out, pending_cursor, out_cursor, state_cursor);
    }
    Goldilocks::Element res = out[(TRANSCRIPT_OUT_SIZE - *out_cursor) % TRANSCRIPT_OUT_SIZE];
    *out_cursor=*out_cursor - 1;
    return res;
}

__global__ void _add(Goldilocks::Element* input, uint64_t size,  Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor)
{
    for (uint64_t i = 0; i < size; i++)
    {
        pending[*pending_cursor] = input[i];
        (*pending_cursor) += 1;
        *out_cursor = 0;
        if (*pending_cursor == TRANSCRIPT_PENDING_SIZE)
        {
            _updateState(state, pending, out, pending_cursor, out_cursor, state_cursor);
        }
    }
}

__global__ void _getField(uint64_t* output, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor)
{
    for (int i = 0; i < 3; i++)
    {
        Goldilocks::Element val = _getFields1(state, pending, out, pending_cursor, out_cursor, state_cursor);
        output[i] = val.fe;
    }
    printf("output GPU: %llu %llu %llu\n", output[0], output[1], output[2]);
   
}

__global__ void __getState(Goldilocks::Element* output, uint64_t nOutputs, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor)
{
    if (*pending_cursor > 0)
    {
        _updateState(state, pending, out, pending_cursor, out_cursor, state_cursor);
    }
    for (int i = 0; i < nOutputs; i++)
    {
        output[i] = state[i];
    }
}

__global__ void __getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor){

    uint64_t totalBits = n * nBits;

    uint64_t NFields = floor((float)(totalBits - 1) / 63) + 1;
    Goldilocks::Element* fields = new Goldilocks::Element[NFields];

    for (uint64_t i = 0; i < NFields; i++)
    {
        fields[i] = _getFields1(state, pending, out, pending_cursor, out_cursor, state_cursor);
    }

    uint64_t curField = 0;
    uint64_t curBit = 0;
    gl64_t* fields_ = (gl64_t*)fields;
    for (uint64_t i = 0; i < n; i++)
    {
        uint64_t a = 0;
        for (uint64_t j = 0; j < nBits; j++)
        {
            uint64_t bit = (uint64_t(fields_[curField]) >> curBit) & 1;
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
    }
    delete[] fields;
}


void TranscriptGL_GPU::init_const()
{
    static int initialized = 0;
    if (initialized == 0)
    {
        initialized = 1;
        
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_C, Poseidon2GoldilocksConstants::C, 118 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(GPU_D, Poseidon2GoldilocksConstants::D, 12 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
       
    }
}

void TranscriptGL_GPU::put(Goldilocks::Element *input, uint64_t size)
{
   _add<<<1,1>>>(input, size, state, pending, out, pending_cursor, out_cursor, state_cursor);
}

void TranscriptGL_GPU::getField(uint64_t* output)
{
    _getField<<<1, 1>>>(output, state, pending, out, pending_cursor, out_cursor, state_cursor);
    
} 

void TranscriptGL_GPU::getState(Goldilocks::Element* output) {
    __getState<<<1, 1>>>(output, TRANSCRIPT_STATE_SIZE, state, pending, out, pending_cursor, out_cursor, state_cursor);
}

void TranscriptGL_GPU::getState(Goldilocks::Element* output, uint64_t nOutputs) {
    __getState<<<1, 1>>>(output, nOutputs, state, pending, out, pending_cursor, out_cursor, state_cursor);
}

void TranscriptGL_GPU::getPermutations(uint64_t *res, uint64_t n, uint64_t nBits)
{
   __getPermutations<<<1, 1>>>(res, n, nBits, state, pending, out, pending_cursor, out_cursor, state_cursor);
}
