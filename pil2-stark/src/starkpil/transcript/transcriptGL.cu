#include "transcriptGL.cuh"
#include "starks_api_internal.hpp"
#ifdef STARK_POSEIDON1
#include "poseidon_goldilocks.cuh"
#else
#include "poseidon2_goldilocks.cuh"
#endif
#include "goldilocks_tooling.cuh"

#include "math.h"

static int initialized = 0;

// Transcript constants — sized for max supported width
#ifdef STARK_POSEIDON1

__device__ __constant__ uint64_t POSEIDON1_GPU_C[150];
__device__ __constant__ uint64_t POSEIDON1_GPU_S[683];
__device__ __constant__ uint64_t POSEIDON1_GPU_M[256];
__device__ __constant__ uint64_t POSEIDON1_GPU_P[256];
#else
__device__ __constant__ gl64_t POSEIDON2_GPU_C[150];
__device__ __constant__ gl64_t POSEIDON2_GPU_D[16]; 
#endif

#ifdef STARK_POSEIDON1
#define TRX_PENDING_SIZE(a) ((uint32_t)(4 * (a)))
#define TRX_OUT_SIZE(a)     ((uint32_t)(4 * ((a) + 1)))
#else
#define TRX_PENDING_SIZE(a) ((uint32_t)(4 * ((a) - 1)))
#define TRX_OUT_SIZE(a)     ((uint32_t)(4 * (a)))
#endif

__device__ void _updateState(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity)
{
    uint32_t transcriptStateSize = HASH_SIZE;
    uint32_t transcriptPendingSize = TRX_PENDING_SIZE(arity);
    uint32_t transcriptOutSize     = TRX_OUT_SIZE(arity);

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
#ifdef STARK_POSEIDON1
    switch(arity) {
        case 2:
            poseidon1PermuteReg<PoseidonGoldilocks<12>::SPONGE_WIDTH,
                                PoseidonGoldilocks<12>::HALF_N_FULL_ROUNDS,
                                PoseidonGoldilocks<12>::N_PARTIAL_ROUNDS,
                                PoseidonGoldilocks<12>::DM>(
                (gl64_t*)out, (gl64_t*)inputs,
                (const gl64_t*)POSEIDON1_GPU_C, (const gl64_t*)POSEIDON1_GPU_S,
                (const gl64_t*)POSEIDON1_GPU_M, (const gl64_t*)POSEIDON1_GPU_P);
            break;
        case 3:
            poseidon1PermuteReg<PoseidonGoldilocks<16>::SPONGE_WIDTH,
                                PoseidonGoldilocks<16>::HALF_N_FULL_ROUNDS,
                                PoseidonGoldilocks<16>::N_PARTIAL_ROUNDS,
                                PoseidonGoldilocks<16>::DM>(
                (gl64_t*)out, (gl64_t*)inputs,
                (const gl64_t*)POSEIDON1_GPU_C, (const gl64_t*)POSEIDON1_GPU_S,
                (const gl64_t*)POSEIDON1_GPU_M, (const gl64_t*)POSEIDON1_GPU_P);
            break;
        default:
            assert(false && "STARK_POSEIDON1 supports arity 2 or 3 only");
            return;
    }
#else
    switch(arity){
        case 2:
            poseidon2PermuteReg<Poseidon2Goldilocks<8>::RATE, Poseidon2Goldilocks<8>::CAPACITY, Poseidon2Goldilocks<8>::SPONGE_WIDTH, Poseidon2Goldilocks<8>::N_FULL_ROUNDS_TOTAL, Poseidon2Goldilocks<8>::N_PARTIAL_ROUNDS, Poseidon2Goldilocks<8>::DM>((gl64_t*)out, (gl64_t*)inputs, POSEIDON2_GPU_C, POSEIDON2_GPU_D);
            break;
        case 3:
            poseidon2PermuteReg<Poseidon2Goldilocks<12>::RATE, Poseidon2Goldilocks<12>::CAPACITY, Poseidon2Goldilocks<12>::SPONGE_WIDTH, Poseidon2Goldilocks<12>::N_FULL_ROUNDS_TOTAL, Poseidon2Goldilocks<12>::N_PARTIAL_ROUNDS, Poseidon2Goldilocks<12>::DM>((gl64_t*)out, (gl64_t*)inputs, POSEIDON2_GPU_C, POSEIDON2_GPU_D);
            break;
        case 4:
            poseidon2PermuteReg<Poseidon2Goldilocks<16>::RATE, Poseidon2Goldilocks<16>::CAPACITY, Poseidon2Goldilocks<16>::SPONGE_WIDTH, Poseidon2Goldilocks<16>::N_FULL_ROUNDS_TOTAL, Poseidon2Goldilocks<16>::N_PARTIAL_ROUNDS, Poseidon2Goldilocks<16>::DM>((gl64_t*)out, (gl64_t*)inputs, POSEIDON2_GPU_C, POSEIDON2_GPU_D);
            break;
        default:
            assert(false && "Unsupported arity");
            return;
    }
#endif

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

__device__ Goldilocks::Element _getFields1(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity){

    uint32_t transcriptOutSize = TRX_OUT_SIZE(arity);

    if (*out_cursor == 0)
    {
        _updateState(state, pending, out, pending_cursor, out_cursor, arity);
    }
    Goldilocks::Element res = out[(transcriptOutSize - *out_cursor) % transcriptOutSize];
    *out_cursor=*out_cursor - 1;
    return res;
}

__global__ void _add(Goldilocks::Element* input, uint64_t size,  Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity)
{
    uint32_t transcriptPendingSize = TRX_PENDING_SIZE(arity);
    for (uint64_t i = 0; i < size; i++)
    {
        pending[*pending_cursor] = input[i];
        (*pending_cursor) += 1;
        *out_cursor = 0;
        if (*pending_cursor == transcriptPendingSize)
        {
            _updateState(state, pending, out, pending_cursor, out_cursor, arity);
        }
    }
}

__global__ void _add2(Goldilocks::Element* input1, uint64_t size1, Goldilocks::Element* input2, uint64_t size2, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity)
{
    uint32_t transcriptPendingSize = TRX_PENDING_SIZE(arity);
    for (uint64_t i = 0; i < size1; i++)
    {
        pending[*pending_cursor] = input1[i];
        (*pending_cursor) += 1;
        *out_cursor = 0;
        if (*pending_cursor == transcriptPendingSize)
        {
            _updateState(state, pending, out, pending_cursor, out_cursor, arity);
        }
    }
    for (uint64_t i = 0; i < size2; i++)
    {
        pending[*pending_cursor] = input2[i];
        (*pending_cursor) += 1;
        *out_cursor = 0;
        if (*pending_cursor == transcriptPendingSize)
        {
            _updateState(state, pending, out, pending_cursor, out_cursor, arity);
        }
    }
}

__global__ void _getField(uint64_t* output, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity)
{
    for (int i = 0; i < 3; i++)
    {
        Goldilocks::Element val = _getFields1(state, pending, out, pending_cursor, out_cursor, arity);
        output[i] = val.fe;
    }
}

__global__ void __getState(Goldilocks::Element* output, uint64_t nOutputs, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity)
{
    if (*pending_cursor > 0)
    {
        _updateState(state, pending, out, pending_cursor, out_cursor, arity);
    }
    for (uint64_t i = 0; i < nOutputs; i++)
    {
        output[i] = state[i];
    }
}

__global__ void __getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity){

    uint64_t totalBits = n * nBits;

    uint64_t NFields = (totalBits + 62) / 63;
    Goldilocks::Element* fields = new Goldilocks::Element[NFields];

    for (uint64_t i = 0; i < NFields; i++)
    {
        fields[i] = _getFields1(state, pending, out, pending_cursor, out_cursor, arity);
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
                a = a + (1ULL << j);
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
        transcriptPendingSize = TRX_PENDING_SIZE(arity);
        transcriptOutSize = TRX_OUT_SIZE(arity);

        CHECKCUDAERR(cudaMalloc((void**)&state, transcriptOutSize * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMalloc((void**)&pending, transcriptPendingSize * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMalloc((void**)&out, transcriptOutSize * sizeof(Goldilocks::Element)));
        CHECKCUDAERR(cudaMalloc((void**)&pending_cursor, sizeof(uint)));
        CHECKCUDAERR(cudaMalloc((void**)&out_cursor, sizeof(uint)));

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
            CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));
#ifdef STARK_POSEIDON1
            if (arity_init == 2) {
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_C, PoseidonGoldilocksConstants::C12, 118 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_M, PoseidonGoldilocksConstants::M12, 144 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_P, PoseidonGoldilocksConstants::P12, 144 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_S, PoseidonGoldilocksConstants::S12, 507 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            } else if (arity_init == 3) {
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_C, PoseidonGoldilocksConstants::C16, 150 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_M, PoseidonGoldilocksConstants::M16, 256 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_P, PoseidonGoldilocksConstants::P16, 256 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_S, PoseidonGoldilocksConstants::S16, 683 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            } else {
                zklog.error("TranscriptGL_GPU::init_const: STARK_POSEIDON1 supports arity 2 or 3 only");
                exitProcess();
                exit(-1);
            }
#else
            if(arity_init == 2) {
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_C, Poseidon2GoldilocksConstants::C8,  (8 * Poseidon2GoldilocksConstants::ROUNDS_F + Poseidon2GoldilocksConstants::ROUNDS_P_8) * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_D, Poseidon2GoldilocksConstants::D8, 8 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            } else if (arity_init == 3) {
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_C, Poseidon2GoldilocksConstants::C12, (12 * Poseidon2GoldilocksConstants::ROUNDS_F + Poseidon2GoldilocksConstants::ROUNDS_P_12) * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_D, Poseidon2GoldilocksConstants::D12, 12 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            } else if (arity_init == 4) {
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_C, Poseidon2GoldilocksConstants::C16, (16 * Poseidon2GoldilocksConstants::ROUNDS_F + Poseidon2GoldilocksConstants::ROUNDS_P_16) * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
                CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_D, Poseidon2GoldilocksConstants::D16, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            } else {
                zklog.error("TranscriptGL_GPU::init_const: Unsupported arity");
                exitProcess();
                exit(-1);
            }
#endif
        }
        initialized = 1;

    }
    cudaSetDevice(deviceId);
}

void TranscriptGL_GPU::put(Goldilocks::Element *input, uint64_t size, cudaStream_t stream)
{
    _add<<<1,1, 0, stream>>>(input, size, state, pending, out, pending_cursor, out_cursor,arity);
}

void TranscriptGL_GPU::put2(Goldilocks::Element *input1, uint64_t size1, Goldilocks::Element *input2, uint64_t size2, cudaStream_t stream)
{
    _add2<<<1,1, 0, stream>>>(input1, size1, input2, size2, state, pending, out, pending_cursor, out_cursor, arity);
}

void TranscriptGL_GPU::getField(uint64_t* output, cudaStream_t stream)
{
    _getField<<<1, 1, 0, stream>>>(output, state, pending, out, pending_cursor, out_cursor, arity);
    
} 

void TranscriptGL_GPU::getState(Goldilocks::Element* output, cudaStream_t stream) {
    __getState<<<1, 1, 0, stream>>>(output, transcriptStateSize, state, pending, out, pending_cursor, out_cursor,arity);
}

void TranscriptGL_GPU::getState(Goldilocks::Element* output, uint64_t nOutputs, cudaStream_t stream) {
    __getState<<<1, 1, 0, stream>>>(output, nOutputs, state, pending, out, pending_cursor, out_cursor,arity);
}

void TranscriptGL_GPU::getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, cudaStream_t stream)
{
    __getPermutations<<<1, 1, 0, stream>>>(res, n, nBits, state, pending, out, pending_cursor, out_cursor, arity);
}
