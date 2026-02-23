#include "transcriptBN128.cuh"
#include "goldilocks_base_field.hpp"
#include "poseidon2/poseidon2_bn128_constants.hpp"
#include "gl64_tooling.cuh"

#include <math.h>

// Track if transcript constants have been initialized
static int transcriptBN128_initialized = 0;

// Macro to declare shared memory arrays for FrElement without dynamic initialization warning
// Uses raw uint32_t storage with reinterpret_cast
#define DECLARE_SHARED_FR_ARRAYS() \
    __shared__ uint32_t _raw_shared_state[32 * 8]; \
    __shared__ uint32_t _raw_tmp[32 * 8]; \
    Poseidon2BN128GPU::FrElement* shared_state = reinterpret_cast<Poseidon2BN128GPU::FrElement*>(_raw_shared_state); \
    Poseidon2BN128GPU::FrElement* tmp = reinterpret_cast<Poseidon2BN128GPU::FrElement*>(_raw_tmp)

// Device constant memory for Poseidon2 C and D constants
__device__ __constant__ Poseidon2BN128GPU::FrElement TRANSCRIPT_GPU_C[185]; // max size for t=16
__device__ __constant__ Poseidon2BN128GPU::FrElement TRANSCRIPT_GPU_D[16];  // max size for t=16

// Round counts for Poseidon2 partial rounds: for t=2,3,4,8,12,16
__device__ __constant__ int TRANSCRIPT_N_ROUNDS_P[6] = {56, 56, 56, 57, 57, 57};

// Map t value to TRANSCRIPT_N_ROUNDS_P index (same as poseidon2_bn128.cu)
__device__ __forceinline__ int transcript_get_nRoundsP_pos(int t) {
    return t <= 4 ? t - 2 : t / 4 + 1;
}


// Device helper: Convert BN128 Fr element and extract limbs
// GPU BN128 elements are stored in Montgomery form as 8 uint32_t limbs
__device__ void fromMontgomeryLimbs(uint64_t* result, Poseidon2BN128GPU::FrElement a) {

    BN128GPUScalarField::fromMontgomery(a);
    // The fr_t structure contains 8 uint32_t values in little-endian order
    // Combine into uint64_t (little-endian)
    result[0] = ((uint64_t)a[1] << 32) | a[0];
    result[1] = ((uint64_t)a[3] << 32) | a[2];
    result[2] = ((uint64_t)a[5] << 32) | a[4];
    result[3] = ((uint64_t)a[7] << 32) | a[6];
}

__device__ void goldilocks_to_fr(Poseidon2BN128GPU::FrElement& result, const Goldilocks::Element& gl) {

    result = BN128GPUScalarField::zero();

    // Reduce from partially reduced form [0, 2*MOD) to canonical form [0, MOD)
    uint64_t val = gl64_reduce(gl);

    // Set the low 64 bits in little-endian limb order
    result[0] = (uint32_t)(val);
    result[1] = (uint32_t)(val >> 32);

    BN128GPUScalarField::toMontgomery(result);
}

// Parallel _updateStateBN128 - uses blockDim.x threads for parallel hash
__device__ void _updateStateBN128(Poseidon2BN128GPU::FrElement* state, Poseidon2BN128GPU::FrElement* pending,
                                  Poseidon2BN128GPU::FrElement* out, uint* pending_cursor, uint* out_cursor,
                                  uint* out3_cursor, uint64_t arity,
                                  Poseidon2BN128GPU::FrElement* shared_state, Poseidon2BN128GPU::FrElement* tmp)
{
    int tid = threadIdx.x;
    int rate = arity - 1;
    int t = arity;

    if (tid == 0) {
        while (*pending_cursor < (uint)rate) {
            pending[*pending_cursor] = BN128GPUScalarField::zero();
            (*pending_cursor)++;
        }

        out[0] = state[0];
        for (int i = 0; i < rate; i++) {
            out[1 + i] = pending[i];
        }
    }
    __syncthreads();

    // Copy out to shared_state for parallel hash
    if (tid < t) {
        shared_state[tid] = out[tid];
    }
    __syncthreads();

    const int nRoundsP = TRANSCRIPT_N_ROUNDS_P[transcript_get_nRoundsP_pos(t)];

    // Call the Poseidon2 parallel hash function
    Poseidon2BN128GPU poseidon2;
    poseidon2.hash_parallel_(shared_state, tmp, t, TRANSCRIPT_GPU_C, TRANSCRIPT_GPU_D, nRoundsP);

    // Copy result back to out
    if (tid < t) {
        out[tid] = shared_state[tid];
    }
    __syncthreads();

    // Thread 0 updates state and cursors
    if (tid == 0) {
        state[0] = out[0];
        *pending_cursor = 0;
        *out_cursor = t;
        *out3_cursor = 0;
    }
    __syncthreads();
}

// Parallel _add1BN128 - must be called with 32 threads
__device__ void _add1BN128(Poseidon2BN128GPU::FrElement* state, Poseidon2BN128GPU::FrElement* pending,
                           Poseidon2BN128GPU::FrElement* out, uint* pending_cursor, uint* out_cursor,
                           uint* out3_cursor, uint64_t arity, const Poseidon2BN128GPU::FrElement& val,
                           Poseidon2BN128GPU::FrElement* shared_state, Poseidon2BN128GPU::FrElement* tmp)
{
    int tid = threadIdx.x;
    __shared__ bool need_hash;
    uint64_t rate = arity - 1;

    // Only thread 0 updates pending and checks if hash is needed
    if (tid == 0) {
        pending[*pending_cursor] = val;
        (*pending_cursor)++;
        *out_cursor = 0;
        need_hash = (*pending_cursor == rate);
    }
    __syncthreads();

    if (need_hash) {
        _updateStateBN128(state, pending, out, pending_cursor, out_cursor, out3_cursor, arity, shared_state, tmp);
    }
}

// Kernel for adding Goldilocks elements - uses 32 threads
__global__ void _addBN128_GL(Goldilocks::Element* input, uint64_t size, Poseidon2BN128GPU::FrElement* state,
                             Poseidon2BN128GPU::FrElement* pending, Poseidon2BN128GPU::FrElement* out,
                             uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity)
{
    DECLARE_SHARED_FR_ARRAYS();

    for (uint64_t i = 0; i < size; i++)
    {
        Poseidon2BN128GPU::FrElement fr_val;
        // All threads convert (redundant but avoids sync issues)
        goldilocks_to_fr(fr_val, input[i]);
        __syncthreads();

        _add1BN128(state, pending, out, pending_cursor, out_cursor, out3_cursor, arity, fr_val, shared_state, tmp);
    }
}

// Kernel for adding Fr elements - uses 32 threads
__global__ void _addBN128_Fr(Poseidon2BN128GPU::FrElement* input, uint64_t size, Poseidon2BN128GPU::FrElement* state,
                             Poseidon2BN128GPU::FrElement* pending, Poseidon2BN128GPU::FrElement* out,
                             uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity)
{
    DECLARE_SHARED_FR_ARRAYS();

    for (uint64_t i = 0; i < size; i++)
    {
        _add1BN128(state, pending, out, pending_cursor, out_cursor, out3_cursor, arity, input[i], shared_state, tmp);
    }
}

// _getFields253 - must be called with 32 threads
__device__ Poseidon2BN128GPU::FrElement _getFields253(Poseidon2BN128GPU::FrElement* state, Poseidon2BN128GPU::FrElement* pending,
                                                     Poseidon2BN128GPU::FrElement* out, uint* pending_cursor, uint* out_cursor,
                                                     uint* out3_cursor, uint64_t arity,
                                                     Poseidon2BN128GPU::FrElement* shared_state, Poseidon2BN128GPU::FrElement* tmp)
{
    int tid = threadIdx.x;
    __shared__ bool found_result;

    while (true) {
        if (tid == 0) {
            found_result = false;
            if (*out_cursor > 0) {
                // Return from out buffer (FIFO - take from front)
                // We use 31th element for output as t will be at most 16
                shared_state[31] = out[0];

                // Shift remaining elements
                for (uint i = 1; i < *out_cursor; i++) {
                    out[i - 1] = out[i];
                }
                (*out_cursor)--;
                found_result = true;
            }
        }
        __syncthreads();

        if (found_result) {
            return shared_state[31];
        }

        // No available output, update state with parallel hash
        _updateStateBN128(state, pending, out, pending_cursor, out_cursor, out3_cursor, arity, shared_state, tmp);
    }
}

// _getFields1BN128 - must be called with 32 threads
__device__ uint64_t _getFields1BN128(Poseidon2BN128GPU::FrElement* state, Poseidon2BN128GPU::FrElement* pending,
                                     Poseidon2BN128GPU::FrElement* out, uint64_t* out3, uint* pending_cursor,
                                     uint* out_cursor, uint* out3_cursor, uint64_t arity,
                                     Poseidon2BN128GPU::FrElement* shared_state, Poseidon2BN128GPU::FrElement* tmp)
{
    int tid = threadIdx.x;
    __shared__ uint64_t result;
    __shared__ bool found_result;

    while (true) {
        // First try to get from out3 buffer
        if (tid == 0) {
            found_result = false;
            if (*out3_cursor > 0) {
                result = out3[0];
                // Shift remaining elements
                for (uint i = 1; i < *out3_cursor; i++) {
                    out3[i - 1] = out3[i];
                }
                (*out3_cursor)--;
                found_result = true;
            }
        }
        __syncthreads();

        if (found_result) {
            return result;
        }

        // Try to get from out buffer and populate out3
        if (tid == 0) {
            if (*out_cursor > 0) {
                Poseidon2BN128GPU::FrElement res = out[0];

                // Shift remaining elements in out
                for (uint i = 1; i < *out_cursor; i++) {
                    out[i - 1] = out[i];
                }
                (*out_cursor)--;

                // Convert from Montgomery and extract uint64_t values
                uint64_t limbs[4];
                fromMontgomeryLimbs(limbs, res);

                out3[0] = limbs[0];
                out3[1] = limbs[1];
                out3[2] = limbs[2];
                *out3_cursor = 3;
            }
        }
        __syncthreads();

        // Check if out3 now has data (we just populated it)
        if (*out3_cursor > 0) {
            continue;  // Go back to get from out3
        }

        // No available output, update state with parallel hash
        _updateStateBN128(state, pending, out, pending_cursor, out_cursor, out3_cursor, arity, shared_state, tmp);
    }
}

// Kernel for getField - uses 32 threads
__global__ void _getFieldBN128(uint64_t* output, Poseidon2BN128GPU::FrElement* state, Poseidon2BN128GPU::FrElement* pending,
                               Poseidon2BN128GPU::FrElement* out, uint64_t* out3, uint* pending_cursor, uint* out_cursor,
                               uint* out3_cursor, uint64_t arity)
{
    DECLARE_SHARED_FR_ARRAYS();

    int tid = threadIdx.x;

    for (int i = 0; i < 3; i++)
    {
        uint64_t val = _getFields1BN128(state, pending, out, out3, pending_cursor, out_cursor, out3_cursor, arity, shared_state, tmp);
        if (tid == 0) {
            output[i] = val;
        }
        __syncthreads();
    }
}

// Kernel for getState - uses 32 threads
__global__ void __getStateBN128(Poseidon2BN128GPU::FrElement* output, Poseidon2BN128GPU::FrElement* state,
                                Poseidon2BN128GPU::FrElement* pending, Poseidon2BN128GPU::FrElement* out,
                                uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity)
{
    DECLARE_SHARED_FR_ARRAYS();

    int tid = threadIdx.x;

    // Check if we need to flush pending
    uint cursor_val = *pending_cursor;
    __syncthreads();

    if (cursor_val > 0) {
        _updateStateBN128(state, pending, out, pending_cursor, out_cursor, out3_cursor, arity, shared_state, tmp);
    }

    if (tid == 0) {
        output[0] = state[0];
    }
}

// Kernel for getPermutations - uses 32 threads
__global__ void __getPermutationsBN128(uint64_t *res, uint64_t n, uint64_t nBits, Poseidon2BN128GPU::FrElement* state,
                                       Poseidon2BN128GPU::FrElement* pending, Poseidon2BN128GPU::FrElement* out,
                                       uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity)
{
    DECLARE_SHARED_FR_ARRAYS();

    int tid = threadIdx.x;

    uint64_t totalBits = n * nBits;
    uint64_t NFields = (totalBits + 252) / 253;  // ceil((totalBits) / 253)

    // Only thread 0 manages the fields array
    Poseidon2BN128GPU::FrElement* fields = nullptr;
    if (tid == 0) {
        fields = new Poseidon2BN128GPU::FrElement[NFields];
    }
    __syncthreads();

    // Broadcast fields pointer to all threads via shared memory
    __shared__ Poseidon2BN128GPU::FrElement* shared_fields;
    if (tid == 0) shared_fields = fields;
    __syncthreads();
    fields = shared_fields;

    for (uint64_t i = 0; i < NFields; i++) {
        Poseidon2BN128GPU::FrElement field_val = _getFields253(state, pending, out, pending_cursor, out_cursor, out3_cursor, arity, shared_state, tmp);
        if (tid == 0) {
            fields[i] = field_val;
        }
        __syncthreads();
    }

    // Only thread 0 does the bit extraction
    if (tid == 0) {
        uint64_t curField = 0;
        uint64_t curBit = 0;

        for (uint64_t i = 0; i < n; i++)
        {
            uint64_t a = 0;
            for (uint64_t j = 0; j < nBits; j++)
            {
                uint64_t limbs[4];
                fromMontgomeryLimbs(limbs, fields[curField]);

                uint64_t limbIdx = curBit / 64;
                uint64_t bitInLimb = curBit % 64;

                uint64_t bit = (limbs[limbIdx] >> bitInLimb) & 1;

                if (bit) {
                    a = a + (1ULL << j);
                }

                curBit++;
                if (curBit == 253) {
                    curBit = 0;
                    curField++;
                }
            }
            res[i] = a;
        }
        delete[] fields;
    }
}

// Constructor — compressor mode: t = arity, rate = arity - 1
TranscriptBN128_GPU::TranscriptBN128_GPU(uint64_t arity, cudaStream_t stream)
{
    this->arity = arity;
    uint64_t rate = arity - 1;

    CHECKCUDAERR(cudaMalloc((void**)&state, sizeof(Poseidon2BN128GPU::FrElement)));
    CHECKCUDAERR(cudaMalloc((void**)&pending, rate * sizeof(Poseidon2BN128GPU::FrElement)));
    CHECKCUDAERR(cudaMalloc((void**)&out, arity * sizeof(Poseidon2BN128GPU::FrElement)));
    CHECKCUDAERR(cudaMalloc((void**)&out3, 3 * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&pending_cursor, sizeof(uint)));
    CHECKCUDAERR(cudaMalloc((void**)&out_cursor, sizeof(uint)));
    CHECKCUDAERR(cudaMalloc((void**)&out3_cursor, sizeof(uint)));

    reset(stream);
}

void TranscriptBN128_GPU::reset(cudaStream_t stream)
{
    uint64_t rate = arity - 1;
    // Initialize state to zero
    CHECKCUDAERR(cudaMemsetAsync(state, 0, sizeof(Poseidon2BN128GPU::FrElement), stream));
    CHECKCUDAERR(cudaMemsetAsync(pending, 0, rate * sizeof(Poseidon2BN128GPU::FrElement), stream));
    CHECKCUDAERR(cudaMemsetAsync(out, 0, arity * sizeof(Poseidon2BN128GPU::FrElement), stream));
    CHECKCUDAERR(cudaMemsetAsync(out3, 0, 3 * sizeof(uint64_t), stream));
    CHECKCUDAERR(cudaMemsetAsync(pending_cursor, 0, sizeof(uint), stream));
    CHECKCUDAERR(cudaMemsetAsync(out_cursor, 0, sizeof(uint), stream));
    CHECKCUDAERR(cudaMemsetAsync(out3_cursor, 0, sizeof(uint), stream));
}

// Helper macro for initializing Poseidon2 constants for a single t value
#define INIT_TRANSCRIPT_P2_CONSTANTS(t_val) do { \
    CHECKCUDAERR(cudaMemcpyToSymbol(TRANSCRIPT_GPU_C, Poseidon2BN128Constants::C##t_val, sizeof(Poseidon2BN128Constants::C##t_val), 0, cudaMemcpyHostToDevice)); \
    CHECKCUDAERR(cudaMemcpyToSymbol(TRANSCRIPT_GPU_D, Poseidon2BN128Constants::D##t_val, sizeof(Poseidon2BN128Constants::D##t_val), 0, cudaMemcpyHostToDevice)); \
} while(0)

void TranscriptBN128_GPU::init_const(uint32_t* gpu_ids, uint32_t num_gpu_ids, uint32_t arity)
{
    if (transcriptBN128_initialized) return;

    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));

    // Poseidon2 compressor mode: t = arity
    uint32_t t = arity;

    for (uint32_t i = 0; i < num_gpu_ids; i++)
    {
        CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));

        switch(t) {
            case 2:  INIT_TRANSCRIPT_P2_CONSTANTS(2);  break;
            case 3:  INIT_TRANSCRIPT_P2_CONSTANTS(3);  break;
            case 4:  INIT_TRANSCRIPT_P2_CONSTANTS(4);  break;
            case 8:  INIT_TRANSCRIPT_P2_CONSTANTS(8);  break;
            case 12: INIT_TRANSCRIPT_P2_CONSTANTS(12); break;
            case 16: INIT_TRANSCRIPT_P2_CONSTANTS(16); break;
            default:
                zklog.error("TranscriptBN128_GPU::init_const: Unsupported t value: " + std::to_string(t));
                exitProcess();
                exit(-1);
        }
    }

    CHECKCUDAERR(cudaSetDevice(deviceId));
    transcriptBN128_initialized = 1;
}

#undef INIT_TRANSCRIPT_P2_CONSTANTS

void TranscriptBN128_GPU::put(Goldilocks::Element *input, uint64_t size, cudaStream_t stream, TimerGPU *timer)
{
    if(timer != nullptr) TimerStartCategoryGPU((*timer), TRANSCRIPT_PUT);
    _addBN128_GL<<<1, 32, 0, stream>>>(input, size, state, pending, out, pending_cursor, out_cursor, out3_cursor, arity);
    if(timer != nullptr) TimerStopCategoryGPU((*timer), TRANSCRIPT_PUT);
}

void TranscriptBN128_GPU::put(Poseidon2BN128GPU::FrElement *input, uint64_t size, cudaStream_t stream, TimerGPU *timer)
{
    if(timer != nullptr) TimerStartCategoryGPU((*timer), TRANSCRIPT_PUT);
    _addBN128_Fr<<<1, 32, 0, stream>>>(input, size, state, pending, out, pending_cursor, out_cursor, out3_cursor, arity);
    if(timer != nullptr) TimerStopCategoryGPU((*timer), TRANSCRIPT_PUT);
}

void TranscriptBN128_GPU::getField(uint64_t *output, cudaStream_t stream)
{
    _getFieldBN128<<<1, 32, 0, stream>>>(output, state, pending, out, out3, pending_cursor, out_cursor, out3_cursor, arity);
}

void TranscriptBN128_GPU::getState(Poseidon2BN128GPU::FrElement* output, cudaStream_t stream)
{
    __getStateBN128<<<1, 32, 0, stream>>>(output, state, pending, out, pending_cursor, out_cursor, out3_cursor, arity);
}

void TranscriptBN128_GPU::getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, cudaStream_t stream)
{
    __getPermutationsBN128<<<1, 32, 0, stream>>>(res, n, nBits, state, pending, out, pending_cursor, out_cursor, out3_cursor, arity);
}
