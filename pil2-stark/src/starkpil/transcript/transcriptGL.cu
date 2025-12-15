#include "transcriptGL.cuh"
#include "poseidon2_goldilocks.cuh"
#include "gl64_tooling.cuh"

#include "math.h"

static int initialized = 0;

__device__ __constant__ gl64_t POSEIDON2_GPU_C[150]; // we allocate the masimum possible size
__device__ __constant__ gl64_t POSEIDON2_GPU_D[16];  // we allocate the masimum possible size

__device__ void _updateState(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor, uint32_t arity) 
{
    uint32_t transcriptStateSize = HASH_SIZE;
    uint32_t transcriptPendingSize = 4 * (arity - 1);
    uint32_t transcriptOutSize = 4 * arity;    
    
    while(*pending_cursor < transcriptPendingSize) {
        pending[*pending_cursor].fe = 0;
        (*pending_cursor) += 1;
    }
    Goldilocks::Element inputs[16]; //max possible size
    for (int i = 0; i < transcriptPendingSize; i++)
    {
        inputs[i] = pending[i];
    }
    for (int i = 0; i < transcriptStateSize; i++)
    {
        inputs[i + transcriptPendingSize] = state[i];
    }
    switch(arity){
        case 2:
            poseidon2_hash_shared<Poseidon2Goldilocks<8>::RATE, Poseidon2Goldilocks<8>::CAPACITY, Poseidon2Goldilocks<8>::SPONGE_WIDTH, Poseidon2Goldilocks<8>::N_FULL_ROUNDS_TOTAL, Poseidon2Goldilocks<8>::N_PARTIAL_ROUNDS>((gl64_t*)out, (gl64_t*)inputs, POSEIDON2_GPU_C, POSEIDON2_GPU_D);
            break;
        case 3:
            poseidon2_hash_shared<Poseidon2Goldilocks<12>::RATE, Poseidon2Goldilocks<12>::CAPACITY, Poseidon2Goldilocks<12>::SPONGE_WIDTH, Poseidon2Goldilocks<12>::N_FULL_ROUNDS_TOTAL, Poseidon2Goldilocks<12>::N_PARTIAL_ROUNDS>((gl64_t*)out, (gl64_t*)inputs, POSEIDON2_GPU_C, POSEIDON2_GPU_D);
            break;
        case 4:
            poseidon2_hash_shared<Poseidon2Goldilocks<16>::RATE, Poseidon2Goldilocks<16>::CAPACITY, Poseidon2Goldilocks<16>::SPONGE_WIDTH, Poseidon2Goldilocks<16>::N_FULL_ROUNDS_TOTAL, Poseidon2Goldilocks<16>::N_PARTIAL_ROUNDS>((gl64_t*)out, (gl64_t*)inputs, POSEIDON2_GPU_C, POSEIDON2_GPU_D);
            break;
        default:
            assert(false && "Unsupported arity");
            return;
    }

    *out_cursor = transcriptOutSize;
    for (int i = 0; i < transcriptPendingSize; i++)
    {
        pending[i].fe = 0;
    }
    *pending_cursor = 0;
    for (int i = 0; i < transcriptOutSize; i++)
    {
        state[i] = out[i];
    }
}

__device__ Goldilocks::Element _getFields1(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor, uint32_t arity){

    uint32_t transcriptOutSize = 4 * arity;

    if (*out_cursor == 0)
    {
        _updateState(state, pending, out, pending_cursor, out_cursor, state_cursor, arity);
    }
    Goldilocks::Element res = out[(transcriptOutSize - *out_cursor) % transcriptOutSize];
    *out_cursor=*out_cursor - 1;
    return res;
}

__global__ void _add(Goldilocks::Element* input, uint64_t size,  Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor, uint32_t arity)
{
    uint32_t transcriptPendingSize = 4 * (arity - 1);
    for (uint64_t i = 0; i < size; i++)
    {
        pending[*pending_cursor] = input[i];
        (*pending_cursor) += 1;
        *out_cursor = 0;
        if (*pending_cursor == transcriptPendingSize)
        {
            _updateState(state, pending, out, pending_cursor, out_cursor, state_cursor, arity);
        }
    }
}

__global__ void _getField(uint64_t* output, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor, uint32_t arity)
{
    for (int i = 0; i < 3; i++)
    {
        Goldilocks::Element val = _getFields1(state, pending, out, pending_cursor, out_cursor, state_cursor, arity);
        output[i] = val.fe;
    }
   
}

__global__ void __getState(Goldilocks::Element* output, uint64_t nOutputs, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor, uint32_t arity)
{
    if (*pending_cursor > 0)
    {
        _updateState(state, pending, out, pending_cursor, out_cursor, state_cursor, arity);
    }
    for (int i = 0; i < nOutputs; i++)
    {
        output[i] = state[i];
    }
}

__global__ void __getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint* state_cursor, uint32_t arity){

    uint64_t totalBits = n * nBits;

    uint64_t NFields = floor((float)(totalBits - 1) / 63) + 1;
    Goldilocks::Element* fields = new Goldilocks::Element[NFields];

    for (uint64_t i = 0; i < NFields; i++)
    {
        fields[i] = _getFields1(state, pending, out, pending_cursor, out_cursor, state_cursor, arity);
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


TranscriptGL_GPU::TranscriptGL_GPU(uint64_t arity, bool custom, cudaStream_t stream){

        this->arity = arity;
        transcriptStateSize = HASH_SIZE;
        transcriptPendingSize = 4*(arity - 1);
        transcriptOutSize = 4*arity;

        CHECKCUDAERR(cudaMalloc((void**)&state, transcriptOutSize * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMalloc((void**)&pending, transcriptPendingSize * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMalloc((void**)&out, transcriptOutSize * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMalloc((void**)&pending_cursor, sizeof(uint)));
        CHECKCUDAERR(cudaMalloc((void**)&out_cursor, sizeof(uint)));
        CHECKCUDAERR(cudaMalloc((void**)&state_cursor, sizeof(uint)));

        reset(stream);
}


void TranscriptGL_GPU::init_const(uint32_t* gpu_ids, uint32_t num_gpu_ids, uint32_t arity_init)
{

    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));
    if (initialized == 0)
    {
        for(int i = 0; i < num_gpu_ids; i++)
        {     
            if(arity_init == 2) {
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_C, Poseidon2GoldilocksConstants::C8,  (8 * Poseidon2GoldilocksConstants::ROUNDS_F + Poseidon2GoldilocksConstants::ROUNDS_P_8) * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_D, Poseidon2GoldilocksConstants::D8, 8 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            } else if (arity_init == 3) {
                CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_C, Poseidon2GoldilocksConstants::C12, (12 * Poseidon2GoldilocksConstants::ROUNDS_F + Poseidon2GoldilocksConstants::ROUNDS_P_12) * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_D, Poseidon2GoldilocksConstants::D12, 12 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            } else if (arity_init == 4) {
                CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_C, Poseidon2GoldilocksConstants::C16, (16 * Poseidon2GoldilocksConstants::ROUNDS_F + Poseidon2GoldilocksConstants::ROUNDS_P_16) * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_D, Poseidon2GoldilocksConstants::D16, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            }else {
                zklog.error("MerkleTreeGL::calculateRootFromProof: Unsupported arity");
                exitProcess();
                exit(-1);
            }
        
        }   
        initialized = 1;
       
    }
    cudaSetDevice(deviceId);
}

void TranscriptGL_GPU::put(Goldilocks::Element *input, uint64_t size, cudaStream_t stream)
{
    size_t sharedMem = (arity*4) * sizeof(gl64_t);
    _add<<<1,1, sharedMem, stream>>>(input, size, state, pending, out, pending_cursor, out_cursor, state_cursor,arity);
}

void TranscriptGL_GPU::getField(uint64_t* output, cudaStream_t stream)
{
    size_t sharedMem = (arity*4) * sizeof(gl64_t);
    _getField<<<1, 1, sharedMem, stream>>>(output, state, pending, out, pending_cursor, out_cursor, state_cursor, arity);
    
} 

void TranscriptGL_GPU::getState(Goldilocks::Element* output, cudaStream_t stream) {
    size_t sharedMem = (arity*4) * sizeof(gl64_t);
    __getState<<<1, 1, sharedMem, stream>>>(output, transcriptStateSize, state, pending, out, pending_cursor, out_cursor, state_cursor,arity);
}

void TranscriptGL_GPU::getState(Goldilocks::Element* output, uint64_t nOutputs, cudaStream_t stream) {
    size_t sharedMem = (arity*4) * sizeof(gl64_t);
    __getState<<<1, 1, sharedMem, stream>>>(output, nOutputs, state, pending, out, pending_cursor, out_cursor, state_cursor,arity);
}

void TranscriptGL_GPU::getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, cudaStream_t stream)
{
    size_t sharedMem = (arity*4) * sizeof(gl64_t);
    __getPermutations<<<1, 1, sharedMem, stream>>>(res, n, nBits, state, pending, out, pending_cursor, out_cursor, state_cursor, arity);
}
