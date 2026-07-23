#include "transcriptGL.cuh"
#include "starks_api_internal.hpp"
#include "poseidon_goldilocks.cuh"
#include "poseidon2_goldilocks.cuh"
#include "goldilocks_tooling.cuh"

#include "math.h"


static int initialized = 0;

// Transcript constants — sized for max supported width. Both families' symbols
// are defined unconditionally
__device__ __constant__ uint64_t POSEIDON1_GPU_C[150];
__device__ __constant__ uint64_t POSEIDON1_GPU_S[683];
__device__ __constant__ uint64_t POSEIDON1_GPU_M[256];
__device__ __constant__ uint64_t POSEIDON1_GPU_P[256];
__device__ __constant__ gl64_t POSEIDON2_GPU_C[150];
__device__ __constant__ gl64_t POSEIDON2_GPU_D[16];

#define TRX_PENDING_SIZE(a) ((uint32_t)(4 * ((a) - 1)))
#define TRX_OUT_SIZE(a)     ((uint32_t)(4 * (a)))

__device__ void _updateState(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily)
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
    if (hashFamily == 1) {
        switch(arity) {
            case 2:
                poseidon1PermuteReg<PoseidonGoldilocks<8>::SPONGE_WIDTH,
                                    PoseidonGoldilocks<8>::HALF_N_FULL_ROUNDS,
                                    PoseidonGoldilocks<8>::N_PARTIAL_ROUNDS>(
                    (gl64_t*)out, (gl64_t*)inputs,
                    (const gl64_t*)POSEIDON1_GPU_C, (const gl64_t*)POSEIDON1_GPU_S,
                    (const gl64_t*)POSEIDON1_GPU_M, (const gl64_t*)POSEIDON1_GPU_P);
                break;
            case 3:
                poseidon1PermuteReg<PoseidonGoldilocks<12>::SPONGE_WIDTH,
                                    PoseidonGoldilocks<12>::HALF_N_FULL_ROUNDS,
                                    PoseidonGoldilocks<12>::N_PARTIAL_ROUNDS>(
                    (gl64_t*)out, (gl64_t*)inputs,
                    (const gl64_t*)POSEIDON1_GPU_C, (const gl64_t*)POSEIDON1_GPU_S,
                    (const gl64_t*)POSEIDON1_GPU_M, (const gl64_t*)POSEIDON1_GPU_P);
                break;
            case 4:
                poseidon1PermuteReg<PoseidonGoldilocks<16>::SPONGE_WIDTH,
                                    PoseidonGoldilocks<16>::HALF_N_FULL_ROUNDS,
                                    PoseidonGoldilocks<16>::N_PARTIAL_ROUNDS>(
                    (gl64_t*)out, (gl64_t*)inputs,
                    (const gl64_t*)POSEIDON1_GPU_C, (const gl64_t*)POSEIDON1_GPU_S,
                    (const gl64_t*)POSEIDON1_GPU_M, (const gl64_t*)POSEIDON1_GPU_P);
                break;
            default:
                assert(false && "Poseidon1 supports arity 2, 3 or 4");
                return;
        }
    } else {
        switch(arity){
            case 2:
                poseidon2PermuteReg<Poseidon2Goldilocks<8>::RATE, Poseidon2Goldilocks<8>::CAPACITY, Poseidon2Goldilocks<8>::SPONGE_WIDTH, Poseidon2Goldilocks<8>::N_FULL_ROUNDS_TOTAL, Poseidon2Goldilocks<8>::N_PARTIAL_ROUNDS>((gl64_t*)out, (gl64_t*)inputs, POSEIDON2_GPU_C, POSEIDON2_GPU_D);
                break;
            case 3:
                poseidon2PermuteReg<Poseidon2Goldilocks<12>::RATE, Poseidon2Goldilocks<12>::CAPACITY, Poseidon2Goldilocks<12>::SPONGE_WIDTH, Poseidon2Goldilocks<12>::N_FULL_ROUNDS_TOTAL, Poseidon2Goldilocks<12>::N_PARTIAL_ROUNDS>((gl64_t*)out, (gl64_t*)inputs, POSEIDON2_GPU_C, POSEIDON2_GPU_D);
                break;
            case 4:
                poseidon2PermuteReg<Poseidon2Goldilocks<16>::RATE, Poseidon2Goldilocks<16>::CAPACITY, Poseidon2Goldilocks<16>::SPONGE_WIDTH, Poseidon2Goldilocks<16>::N_FULL_ROUNDS_TOTAL, Poseidon2Goldilocks<16>::N_PARTIAL_ROUNDS>((gl64_t*)out, (gl64_t*)inputs, POSEIDON2_GPU_C, POSEIDON2_GPU_D);
                break;
            default:
                assert(false && "Poseidon2 supports arity 2, 3 or 4");
                return;
        }
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

__device__ Goldilocks::Element _getFields1(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily){

    uint32_t transcriptOutSize = TRX_OUT_SIZE(arity);

    if (*out_cursor == 0)
    {
        _updateState(state, pending, out, pending_cursor, out_cursor, arity, hashFamily);
    }
    Goldilocks::Element res = out[(transcriptOutSize - *out_cursor) % transcriptOutSize];
    *out_cursor=*out_cursor - 1;
    return res;
}

__global__ void _add(Goldilocks::Element* input, uint64_t size,  Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily)
{
    uint32_t transcriptPendingSize = TRX_PENDING_SIZE(arity);
    for (uint64_t i = 0; i < size; i++)
    {
        pending[*pending_cursor] = input[i];
        (*pending_cursor) += 1;
        *out_cursor = 0;
        if (*pending_cursor == transcriptPendingSize)
        {
            _updateState(state, pending, out, pending_cursor, out_cursor, arity, hashFamily);
        }
    }
}

__global__ void _add2(Goldilocks::Element* input1, uint64_t size1, Goldilocks::Element* input2, uint64_t size2, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily)
{
    uint32_t transcriptPendingSize = TRX_PENDING_SIZE(arity);
    for (uint64_t i = 0; i < size1; i++)
    {
        pending[*pending_cursor] = input1[i];
        (*pending_cursor) += 1;
        *out_cursor = 0;
        if (*pending_cursor == transcriptPendingSize)
        {
            _updateState(state, pending, out, pending_cursor, out_cursor, arity, hashFamily);
        }
    }
    for (uint64_t i = 0; i < size2; i++)
    {
        pending[*pending_cursor] = input2[i];
        (*pending_cursor) += 1;
        *out_cursor = 0;
        if (*pending_cursor == transcriptPendingSize)
        {
            _updateState(state, pending, out, pending_cursor, out_cursor, arity, hashFamily);
        }
    }
}

__global__ void _getField(uint64_t* output, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily)
{
    for (int i = 0; i < 3; i++)
    {
        Goldilocks::Element val = _getFields1(state, pending, out, pending_cursor, out_cursor, arity, hashFamily);
        output[i] = val.fe;
    }
}

__global__ void __getState(Goldilocks::Element* output, uint64_t nOutputs, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily)
{
    if (*pending_cursor > 0)
    {
        _updateState(state, pending, out, pending_cursor, out_cursor, arity, hashFamily);
    }
    for (uint64_t i = 0; i < nOutputs; i++)
    {
        output[i] = state[i];
    }
}

__global__ void __getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, Goldilocks::Element* fields, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily){

    uint64_t totalBits = n * nBits;

    uint64_t NFields = (totalBits + 62) / 63;

    for (uint64_t i = 0; i < NFields; i++)
    {
        fields[i] = _getFields1(state, pending, out, pending_cursor, out_cursor, arity, hashFamily);
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
}


__device__ void _updateStateWarp(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily)
{
    const uint32_t lane = threadIdx.x;
    const uint32_t transcriptPendingSize = TRX_PENDING_SIZE(arity);
    const uint32_t transcriptOutSize     = TRX_OUT_SIZE(arity);  // = SPONGE_WIDTH
    const uint32_t W = transcriptOutSize;

    assert(W < 32 && "transcript warp permute: SPONGE_WIDTH must fit in one warp");

    if (lane == 0) {
        while (*pending_cursor < transcriptPendingSize) {
            pending[*pending_cursor].fe = 0;
            (*pending_cursor) += 1;
        }
    }
    __syncwarp();

    // Each active lane keeps its sponge element [pending | state] in a register.
    gl64_t v;
    if (lane < transcriptPendingSize)
        v = ((gl64_t*)pending)[lane];
    else if (lane < W)
        v = ((gl64_t*)state)[lane - transcriptPendingSize];

    if (lane < W) {
        const uint32_t mask = (1u << W) - 1u;   // active lanes 0..W-1
        if (hashFamily == 1) {
            switch(arity) {
                case 2:
                    v = poseidon1PermuteWarpReg<PoseidonGoldilocks<8>::SPONGE_WIDTH,
                                                PoseidonGoldilocks<8>::HALF_N_FULL_ROUNDS,
                                                PoseidonGoldilocks<8>::N_PARTIAL_ROUNDS>(
                        v, mask, (const gl64_t*)POSEIDON1_GPU_C, (const gl64_t*)POSEIDON1_GPU_S,
                        (const gl64_t*)POSEIDON1_GPU_M, (const gl64_t*)POSEIDON1_GPU_P);
                    break;
                case 3:
                    v = poseidon1PermuteWarpReg<PoseidonGoldilocks<12>::SPONGE_WIDTH,
                                                PoseidonGoldilocks<12>::HALF_N_FULL_ROUNDS,
                                                PoseidonGoldilocks<12>::N_PARTIAL_ROUNDS>(
                        v, mask, (const gl64_t*)POSEIDON1_GPU_C, (const gl64_t*)POSEIDON1_GPU_S,
                        (const gl64_t*)POSEIDON1_GPU_M, (const gl64_t*)POSEIDON1_GPU_P);
                    break;
                case 4:
                    v = poseidon1PermuteWarpReg<PoseidonGoldilocks<16>::SPONGE_WIDTH,
                                                PoseidonGoldilocks<16>::HALF_N_FULL_ROUNDS,
                                                PoseidonGoldilocks<16>::N_PARTIAL_ROUNDS>(
                        v, mask, (const gl64_t*)POSEIDON1_GPU_C, (const gl64_t*)POSEIDON1_GPU_S,
                        (const gl64_t*)POSEIDON1_GPU_M, (const gl64_t*)POSEIDON1_GPU_P);
                    break;
                default:
                    assert(false && "Poseidon1 supports arity 2, 3 or 4");
                    break;
            }
        } else {
            switch(arity){
                case 2:
                    v = poseidon2PermuteWarpReg<Poseidon2Goldilocks<8>::RATE, Poseidon2Goldilocks<8>::CAPACITY, Poseidon2Goldilocks<8>::SPONGE_WIDTH, Poseidon2Goldilocks<8>::N_FULL_ROUNDS_TOTAL, Poseidon2Goldilocks<8>::N_PARTIAL_ROUNDS>(v, mask, POSEIDON2_GPU_C, POSEIDON2_GPU_D);
                    break;
                case 3:
                    v = poseidon2PermuteWarpReg<Poseidon2Goldilocks<12>::RATE, Poseidon2Goldilocks<12>::CAPACITY, Poseidon2Goldilocks<12>::SPONGE_WIDTH, Poseidon2Goldilocks<12>::N_FULL_ROUNDS_TOTAL, Poseidon2Goldilocks<12>::N_PARTIAL_ROUNDS>(v, mask, POSEIDON2_GPU_C, POSEIDON2_GPU_D);
                    break;
                case 4:
                    v = poseidon2PermuteWarpReg<Poseidon2Goldilocks<16>::RATE, Poseidon2Goldilocks<16>::CAPACITY, Poseidon2Goldilocks<16>::SPONGE_WIDTH, Poseidon2Goldilocks<16>::N_FULL_ROUNDS_TOTAL, Poseidon2Goldilocks<16>::N_PARTIAL_ROUNDS>(v, mask, POSEIDON2_GPU_C, POSEIDON2_GPU_D);
                    break;
                default:
                    assert(false && "Poseidon2 supports arity 2, 3 or 4");
                    break;
            }
        }
        // v now holds this lane's permutation output; write back state and out.
        ((gl64_t*)out)[lane]   = v;
        ((gl64_t*)state)[lane] = v;
    }

    if (lane == 0) {
        for (uint32_t i = 0; i < transcriptPendingSize; i++)
            pending[i].fe = 0;
        *pending_cursor = 0;
        *out_cursor = transcriptOutSize;
    }
    __syncwarp();
}

__global__ void _addWarp(Goldilocks::Element* input, uint64_t size, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily)
{
    const uint32_t lane = threadIdx.x;
    const uint32_t transcriptPendingSize = TRX_PENDING_SIZE(arity);
    for (uint64_t i = 0; i < size; i++)
    {
        int full = 0;
        if (lane == 0) {
            pending[*pending_cursor] = input[i];
            (*pending_cursor) += 1;
            *out_cursor = 0;
            full = (*pending_cursor == transcriptPendingSize);
        }
        full = __shfl_sync(0xffffffff, full, 0);
        if (full)
            _updateStateWarp(state, pending, out, pending_cursor, out_cursor, arity, hashFamily);
    }
}

__global__ void _add2Warp(Goldilocks::Element* input1, uint64_t size1, Goldilocks::Element* input2, uint64_t size2, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily)
{
    const uint32_t lane = threadIdx.x;
    const uint32_t transcriptPendingSize = TRX_PENDING_SIZE(arity);
    for (uint64_t i = 0; i < size1; i++)
    {
        int full = 0;
        if (lane == 0) {
            pending[*pending_cursor] = input1[i];
            (*pending_cursor) += 1;
            *out_cursor = 0;
            full = (*pending_cursor == transcriptPendingSize);
        }
        full = __shfl_sync(0xffffffff, full, 0);
        if (full)
            _updateStateWarp(state, pending, out, pending_cursor, out_cursor, arity, hashFamily);
    }
    for (uint64_t i = 0; i < size2; i++)
    {
        int full = 0;
        if (lane == 0) {
            pending[*pending_cursor] = input2[i];
            (*pending_cursor) += 1;
            *out_cursor = 0;
            full = (*pending_cursor == transcriptPendingSize);
        }
        full = __shfl_sync(0xffffffff, full, 0);
        if (full)
            _updateStateWarp(state, pending, out, pending_cursor, out_cursor, arity, hashFamily);
    }
}

__global__ void _getFieldWarp(uint64_t* output, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily)
{
    const uint32_t lane = threadIdx.x;
    const uint32_t transcriptOutSize = TRX_OUT_SIZE(arity);
    for (int i = 0; i < 3; i++)
    {
        int need = 0;
        if (lane == 0) need = (*out_cursor == 0);
        need = __shfl_sync(0xffffffff, need, 0);
        if (need)
            _updateStateWarp(state, pending, out, pending_cursor, out_cursor, arity, hashFamily);
        if (lane == 0) {
            Goldilocks::Element res = out[(transcriptOutSize - *out_cursor) % transcriptOutSize];
            *out_cursor = *out_cursor - 1;
            output[i] = res.fe;
        }
        __syncwarp();
    }
}

__global__ void __getStateWarp(Goldilocks::Element* output, uint64_t nOutputs, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily)
{
    const uint32_t lane = threadIdx.x;
    int need = 0;
    if (lane == 0) need = (*pending_cursor > 0);
    need = __shfl_sync(0xffffffff, need, 0);
    if (need)
        _updateStateWarp(state, pending, out, pending_cursor, out_cursor, arity, hashFamily);
    if (lane == 0) {
        for (uint64_t i = 0; i < nOutputs; i++)
            output[i] = state[i];
    }
}

__global__ void __getPermutationsWarp(uint64_t *res, uint64_t n, uint64_t nBits, Goldilocks::Element* fields, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily)
{
    const uint32_t lane = threadIdx.x;
    const uint32_t transcriptOutSize = TRX_OUT_SIZE(arity);

    uint64_t totalBits = n * nBits;
    uint64_t NFields = (totalBits + 62) / 63;

    for (uint64_t i = 0; i < NFields; i++)
    {
        int need = 0;
        if (lane == 0) need = (*out_cursor == 0);
        need = __shfl_sync(0xffffffff, need, 0);
        if (need)
            _updateStateWarp(state, pending, out, pending_cursor, out_cursor, arity, hashFamily);
        if (lane == 0) {
            fields[i] = out[(transcriptOutSize - *out_cursor) % transcriptOutSize];
            *out_cursor = *out_cursor - 1;
        }
        __syncwarp();
    }

    if (lane == 0) {
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
    }
}


TranscriptGL_GPU::TranscriptGL_GPU(uint64_t arity, bool custom, cudaStream_t stream, bool parallel){

        this->arity = arity;
        this->parallel = parallel;
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
    if (initialized) return;

    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));
    for (uint32_t i = 0; i < num_gpu_ids; i++) {
        CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));
        if (arity_init == 2) {
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_C, PoseidonGoldilocksConstants::C8,  86  * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_M, PoseidonGoldilocksConstants::M8,  64  * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_P, PoseidonGoldilocksConstants::P8,  64  * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_S, PoseidonGoldilocksConstants::S8,  330 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_C, Poseidon2GoldilocksConstants::C8,  (8 * Poseidon2GoldilocksConstants::ROUNDS_F + Poseidon2GoldilocksConstants::ROUNDS_P_8) * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_D, Poseidon2GoldilocksConstants::D8,  8 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
        } else if (arity_init == 3) {
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_C, PoseidonGoldilocksConstants::C12, 118 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_M, PoseidonGoldilocksConstants::M12, 144 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_P, PoseidonGoldilocksConstants::P12, 144 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_S, PoseidonGoldilocksConstants::S12, 507 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_C, Poseidon2GoldilocksConstants::C12, (12 * Poseidon2GoldilocksConstants::ROUNDS_F + Poseidon2GoldilocksConstants::ROUNDS_P_12) * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_D, Poseidon2GoldilocksConstants::D12, 12 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
        } else if (arity_init == 4) {
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_C, PoseidonGoldilocksConstants::C16, 150 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_M, PoseidonGoldilocksConstants::M16, 256 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_P, PoseidonGoldilocksConstants::P16, 256 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON1_GPU_S, PoseidonGoldilocksConstants::S16, 683 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_C, Poseidon2GoldilocksConstants::C16, (16 * Poseidon2GoldilocksConstants::ROUNDS_F + Poseidon2GoldilocksConstants::ROUNDS_P_16) * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
            CHECKCUDAERR(cudaMemcpyToSymbol(POSEIDON2_GPU_D, Poseidon2GoldilocksConstants::D16, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
        } else {
            zklog.error("TranscriptGL_GPU::init_const: supports arity 2, 3 or 4");
            exitProcess();
            exit(-1);
        }
    }
    initialized = 1;
    cudaSetDevice(deviceId);
}

void TranscriptGL_GPU::put(Goldilocks::Element *input, uint64_t size, cudaStream_t stream)
{
    if (parallel)
        _addWarp<<<1, 32, 0, stream>>>(input, size, state, pending, out, pending_cursor, out_cursor, arity, static_cast<uint8_t>(get_hash_family()));
    else
        _add<<<1,1, 0, stream>>>(input, size, state, pending, out, pending_cursor, out_cursor, arity, static_cast<uint8_t>(get_hash_family()));
}

void TranscriptGL_GPU::put2(Goldilocks::Element *input1, uint64_t size1, Goldilocks::Element *input2, uint64_t size2, cudaStream_t stream)
{
    if (parallel)
        _add2Warp<<<1, 32, 0, stream>>>(input1, size1, input2, size2, state, pending, out, pending_cursor, out_cursor, arity, static_cast<uint8_t>(get_hash_family()));
    else
        _add2<<<1,1, 0, stream>>>(input1, size1, input2, size2, state, pending, out, pending_cursor, out_cursor, arity, static_cast<uint8_t>(get_hash_family()));
}

void TranscriptGL_GPU::getField(uint64_t* output, cudaStream_t stream)
{
    if (parallel)
        _getFieldWarp<<<1, 32, 0, stream>>>(output, state, pending, out, pending_cursor, out_cursor, arity, static_cast<uint8_t>(get_hash_family()));
    else
        _getField<<<1, 1, 0, stream>>>(output, state, pending, out, pending_cursor, out_cursor, arity, static_cast<uint8_t>(get_hash_family()));
}

void TranscriptGL_GPU::getState(Goldilocks::Element* output, cudaStream_t stream) {
    if (parallel)
        __getStateWarp<<<1, 32, 0, stream>>>(output, transcriptStateSize, state, pending, out, pending_cursor, out_cursor, arity, static_cast<uint8_t>(get_hash_family()));
    else
        __getState<<<1, 1, 0, stream>>>(output, transcriptStateSize, state, pending, out, pending_cursor, out_cursor, arity, static_cast<uint8_t>(get_hash_family()));
}

void TranscriptGL_GPU::getState(Goldilocks::Element* output, uint64_t nOutputs, cudaStream_t stream) {
    if (parallel)
        __getStateWarp<<<1, 32, 0, stream>>>(output, nOutputs, state, pending, out, pending_cursor, out_cursor, arity, static_cast<uint8_t>(get_hash_family()));
    else
        __getState<<<1, 1, 0, stream>>>(output, nOutputs, state, pending, out, pending_cursor, out_cursor, arity, static_cast<uint8_t>(get_hash_family()));
}

void TranscriptGL_GPU::getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, Goldilocks::Element* perm_scratch, cudaStream_t stream)
{
    if (parallel)
        __getPermutationsWarp<<<1, 32, 0, stream>>>(res, n, nBits, perm_scratch, state, pending, out, pending_cursor, out_cursor, arity, static_cast<uint8_t>(get_hash_family()));
    else
        __getPermutations<<<1, 1, 0, stream>>>(res, n, nBits, perm_scratch, state, pending, out, pending_cursor, out_cursor, arity, static_cast<uint8_t>(get_hash_family()));
}
