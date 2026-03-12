#include "transcriptBN128.cuh"
#include "goldilocks_base_field.hpp"
#include "poseidon/poseidon_bn128_constants.hpp"
#include "gl64_tooling.cuh"

#include <math.h>

// Track if transcript constants have been initialized
static int transcriptBN128_initialized = 0;

// Macro to declare shared memory arrays for FrElement without dynamic initialization warning
// Uses raw uint32_t storage with reinterpret_cast
#define DECLARE_SHARED_FR_ARRAYS() \
    __shared__ uint32_t _raw_shared_state[32 * 8]; \
    __shared__ uint32_t _raw_tmp[32 * 8]; \
    PoseidonBN128GPU::FrElement* shared_state = reinterpret_cast<PoseidonBN128GPU::FrElement*>(_raw_shared_state); \
    PoseidonBN128GPU::FrElement* tmp = reinterpret_cast<PoseidonBN128GPU::FrElement*>(_raw_tmp)

// Device-side pointers to Poseidon constants (indexed by t-2)
__device__ PoseidonBN128GPU::FrElement *TRANSCRIPT_GPU_C_ptr = nullptr;
__device__ PoseidonBN128GPU::FrElement *TRANSCRIPT_GPU_M_ptr = nullptr;
__device__ PoseidonBN128GPU::FrElement *TRANSCRIPT_GPU_P_ptr = nullptr;
__device__ PoseidonBN128GPU::FrElement *TRANSCRIPT_GPU_S_ptr = nullptr;

// Host-side pointers for cleanup
static PoseidonBN128GPU::FrElement *h_TRANSCRIPT_GPU_C = nullptr;
static PoseidonBN128GPU::FrElement *h_TRANSCRIPT_GPU_M = nullptr;
static PoseidonBN128GPU::FrElement *h_TRANSCRIPT_GPU_P = nullptr;
static PoseidonBN128GPU::FrElement *h_TRANSCRIPT_GPU_S = nullptr;

// Round counts for partial rounds (indexed by t-2)
__device__ __constant__ int TRANSCRIPT_N_ROUNDS_P[16] = {56, 57, 56, 60, 60, 63, 64, 63, 60, 66, 60, 65, 70, 60, 64, 68};


// Device helper: Convert BN128 Fr element and extract limbs
// GPU BN128 elements are stored in Montgomery form as 8 uint32_t limbs
__device__ void fromMontgomeryLimbs(uint64_t* result, PoseidonBN128GPU::FrElement a) {
    
    BN128GPUScalarField::fromMontgomery(a);
    // The fr_t structure contains 8 uint32_t values in little-endian order
    // Combine into uint64_t (little-endian)
    result[0] = ((uint64_t)a[1] << 32) | a[0];
    result[1] = ((uint64_t)a[3] << 32) | a[2];
    result[2] = ((uint64_t)a[5] << 32) | a[4];
    result[3] = ((uint64_t)a[7] << 32) | a[6];
}

__device__ void goldilocks_to_fr(PoseidonBN128GPU::FrElement& result, const Goldilocks::Element& gl) {
    
    result = BN128GPUScalarField::zero();
    
    // Reduce from partially reduced form [0, 2*MOD) to canonical form [0, MOD)
    uint64_t val = gl64_reduce(gl);
    
    // Set the low 64 bits in little-endian limb order
    result[0] = (uint32_t)(val);
    result[1] = (uint32_t)(val >> 32);
    
    BN128GPUScalarField::toMontgomery(result);
}

// Parallel _updateStateBN128 - uses blockDim.x threads for parallel hash
// shared_state and tmp must be shared memory arrays of size >= t (t = arity + 1)
__device__ void _updateStateBN128(PoseidonBN128GPU::FrElement* state, PoseidonBN128GPU::FrElement* pending, 
                                  PoseidonBN128GPU::FrElement* out, uint* pending_cursor, uint* out_cursor, 
                                  uint* out3_cursor, uint64_t arity,
                                  PoseidonBN128GPU::FrElement* shared_state, PoseidonBN128GPU::FrElement* tmp) 
{
    int tid = threadIdx.x;
    int t = 1 + arity;
    
    if (tid == 0) {
        while (*pending_cursor < arity) {
            pending[*pending_cursor] = BN128GPUScalarField::zero();
            (*pending_cursor)++;
        }
        
        out[0] = state[0];
        for (uint64_t i = 0; i < arity; i++) {
            out[1 + i] = pending[i];
        }
    }
    __syncthreads();
    
    // Copy out to shared_state for parallel hash
    if (tid < t) {
        shared_state[tid] = out[tid];
    }
    __syncthreads();
    
    const int nRoundsP = TRANSCRIPT_N_ROUNDS_P[1 + arity - 2];
    
    // Call the parallel hash function
    PoseidonBN128GPU poseidon;
    poseidon.hash_parallel_(shared_state, tmp, t, TRANSCRIPT_GPU_C_ptr, TRANSCRIPT_GPU_M_ptr, TRANSCRIPT_GPU_P_ptr, TRANSCRIPT_GPU_S_ptr, nRoundsP);
    
    // Copy result back to out
    if (tid < t) {
        out[tid] = shared_state[tid];
    }
    __syncthreads();
    
    // Thread 0 updates state and cursors
    if (tid == 0) {
        state[0] = out[0];
        *pending_cursor = 0;    
        *out_cursor = 1 + arity;
        *out3_cursor = 0;
    }
    __syncthreads();
}

// Parallel _add1BN128 - must be called with 32 threads
__device__ void _add1BN128(PoseidonBN128GPU::FrElement* state, PoseidonBN128GPU::FrElement* pending, 
                           PoseidonBN128GPU::FrElement* out, uint* pending_cursor, uint* out_cursor, 
                           uint* out3_cursor, uint64_t arity, const PoseidonBN128GPU::FrElement& val,
                           PoseidonBN128GPU::FrElement* shared_state, PoseidonBN128GPU::FrElement* tmp)
{
    int tid = threadIdx.x;
    __shared__ bool need_hash;
    
    // Only thread 0 updates pending and checks if hash is needed
    if (tid == 0) {
        pending[*pending_cursor] = val;
        (*pending_cursor)++;
        *out_cursor = 0;
        need_hash = (*pending_cursor == arity);
    }
    __syncthreads();
    
    if (need_hash) {
        _updateStateBN128(state, pending, out, pending_cursor, out_cursor, out3_cursor, arity, shared_state, tmp);
    }
}

// Kernel for adding Goldilocks elements - uses 32 threads
__global__ void _addBN128_GL(Goldilocks::Element* input, uint64_t size, PoseidonBN128GPU::FrElement* state, 
                             PoseidonBN128GPU::FrElement* pending, PoseidonBN128GPU::FrElement* out, 
                             uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity)
{
    DECLARE_SHARED_FR_ARRAYS();
    
    for (uint64_t i = 0; i < size; i++)
    {
        PoseidonBN128GPU::FrElement fr_val;
        // All threads convert (redundant but avoids sync issues)
        goldilocks_to_fr(fr_val, input[i]);
        __syncthreads();
        
        _add1BN128(state, pending, out, pending_cursor, out_cursor, out3_cursor, arity, fr_val, shared_state, tmp);
    }
}

// Kernel for adding Fr elements - uses 32 threads
__global__ void _addBN128_Fr(PoseidonBN128GPU::FrElement* input, uint64_t size, PoseidonBN128GPU::FrElement* state, 
                             PoseidonBN128GPU::FrElement* pending, PoseidonBN128GPU::FrElement* out, 
                             uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity)
{
    DECLARE_SHARED_FR_ARRAYS();
    
    for (uint64_t i = 0; i < size; i++)
    {
        _add1BN128(state, pending, out, pending_cursor, out_cursor, out3_cursor, arity, input[i], shared_state, tmp);
    }
}

// _getFields253 - must be called with 32 threads
__device__ PoseidonBN128GPU::FrElement _getFields253(PoseidonBN128GPU::FrElement* state, PoseidonBN128GPU::FrElement* pending, 
                                                     PoseidonBN128GPU::FrElement* out, uint* pending_cursor, uint* out_cursor, 
                                                     uint* out3_cursor, uint64_t arity,
                                                     PoseidonBN128GPU::FrElement* shared_state, PoseidonBN128GPU::FrElement* tmp) 
{
    int tid = threadIdx.x;
    __shared__ bool found_result;
    
    while (true) {
        if (tid == 0) {
            found_result = false;
            if (*out_cursor > 0) {
                // Return from out buffer (FIFO - take from front)
                // We use 31th element for output as t will be at most 17
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
__device__ uint64_t _getFields1BN128(PoseidonBN128GPU::FrElement* state, PoseidonBN128GPU::FrElement* pending, 
                                     PoseidonBN128GPU::FrElement* out, uint64_t* out3, uint* pending_cursor, 
                                     uint* out_cursor, uint* out3_cursor, uint64_t arity,
                                     PoseidonBN128GPU::FrElement* shared_state, PoseidonBN128GPU::FrElement* tmp)
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
                PoseidonBN128GPU::FrElement res = out[0];
                
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
__global__ void _getFieldBN128(uint64_t* output, PoseidonBN128GPU::FrElement* state, PoseidonBN128GPU::FrElement* pending, 
                               PoseidonBN128GPU::FrElement* out, uint64_t* out3, uint* pending_cursor, uint* out_cursor, 
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
__global__ void __getStateBN128(PoseidonBN128GPU::FrElement* output, PoseidonBN128GPU::FrElement* state, 
                                PoseidonBN128GPU::FrElement* pending, PoseidonBN128GPU::FrElement* out, 
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
__global__ void __getPermutationsBN128(uint64_t *res, uint64_t n, uint64_t nBits, PoseidonBN128GPU::FrElement* state, 
                                       PoseidonBN128GPU::FrElement* pending, PoseidonBN128GPU::FrElement* out, 
                                       uint* pending_cursor, uint* out_cursor, uint* out3_cursor, uint64_t arity)
{
    DECLARE_SHARED_FR_ARRAYS();
    
    int tid = threadIdx.x;
    
    uint64_t totalBits = n * nBits;
    uint64_t NFields = (totalBits + 252) / 253;  // ceil((totalBits) / 253)
    
    // Only thread 0 manages the fields array
    PoseidonBN128GPU::FrElement* fields = nullptr;
    if (tid == 0) {
        fields = new PoseidonBN128GPU::FrElement[NFields];
    }
    __syncthreads();
    
    // Broadcast fields pointer to all threads via shared memory
    __shared__ PoseidonBN128GPU::FrElement* shared_fields;
    if (tid == 0) shared_fields = fields;
    __syncthreads();
    fields = shared_fields;
    
    for (uint64_t i = 0; i < NFields; i++) {
        PoseidonBN128GPU::FrElement field_val = _getFields253(state, pending, out, pending_cursor, out_cursor, out3_cursor, arity, shared_state, tmp);
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

// Constructor
TranscriptBN128_GPU::TranscriptBN128_GPU(uint64_t arity, bool custom, cudaStream_t stream)
{
    this->arity = custom ? arity : 16;
    
    CHECKCUDAERR(cudaMalloc((void**)&state, sizeof(PoseidonBN128GPU::FrElement)));
    CHECKCUDAERR(cudaMalloc((void**)&pending, this->arity * sizeof(PoseidonBN128GPU::FrElement)));
    CHECKCUDAERR(cudaMalloc((void**)&out, (1 + this->arity) * sizeof(PoseidonBN128GPU::FrElement)));
    CHECKCUDAERR(cudaMalloc((void**)&out3, 3 * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&pending_cursor, sizeof(uint)));
    CHECKCUDAERR(cudaMalloc((void**)&out_cursor, sizeof(uint)));
    CHECKCUDAERR(cudaMalloc((void**)&out3_cursor, sizeof(uint)));
    
    reset(stream);
}

void TranscriptBN128_GPU::reset(cudaStream_t stream)
{
    // Initialize state to zero
    CHECKCUDAERR(cudaMemsetAsync(state, 0, sizeof(PoseidonBN128GPU::FrElement), stream));
    CHECKCUDAERR(cudaMemsetAsync(pending, 0, arity * sizeof(PoseidonBN128GPU::FrElement), stream));
    CHECKCUDAERR(cudaMemsetAsync(out, 0, (1 + arity) * sizeof(PoseidonBN128GPU::FrElement), stream));
    CHECKCUDAERR(cudaMemsetAsync(out3, 0, 3 * sizeof(uint64_t), stream));
    CHECKCUDAERR(cudaMemsetAsync(pending_cursor, 0, sizeof(uint), stream));
    CHECKCUDAERR(cudaMemsetAsync(out_cursor, 0, sizeof(uint), stream));
    CHECKCUDAERR(cudaMemsetAsync(out3_cursor, 0, sizeof(uint), stream));
}

// Helper macro for initializing constants for a single t value
#define INIT_TRANSCRIPT_T_CONSTANTS(t_val) do { \
    CHECKCUDAERR(cudaMalloc(&h_TRANSCRIPT_GPU_C, sizeof(PoseidonBN128Constants::C##t_val))); \
    CHECKCUDAERR(cudaMemcpy(h_TRANSCRIPT_GPU_C, PoseidonBN128Constants::C##t_val, sizeof(PoseidonBN128Constants::C##t_val), cudaMemcpyHostToDevice)); \
    CHECKCUDAERR(cudaMemcpyToSymbol(TRANSCRIPT_GPU_C_ptr, &h_TRANSCRIPT_GPU_C, sizeof(PoseidonBN128GPU::FrElement*))); \
    CHECKCUDAERR(cudaMalloc(&h_TRANSCRIPT_GPU_M, sizeof(PoseidonBN128Constants::M##t_val))); \
    CHECKCUDAERR(cudaMemcpy(h_TRANSCRIPT_GPU_M, PoseidonBN128Constants::M##t_val, sizeof(PoseidonBN128Constants::M##t_val), cudaMemcpyHostToDevice)); \
    CHECKCUDAERR(cudaMemcpyToSymbol(TRANSCRIPT_GPU_M_ptr, &h_TRANSCRIPT_GPU_M, sizeof(PoseidonBN128GPU::FrElement*))); \
    CHECKCUDAERR(cudaMalloc(&h_TRANSCRIPT_GPU_P, sizeof(PoseidonBN128Constants::P##t_val))); \
    CHECKCUDAERR(cudaMemcpy(h_TRANSCRIPT_GPU_P, PoseidonBN128Constants::P##t_val, sizeof(PoseidonBN128Constants::P##t_val), cudaMemcpyHostToDevice)); \
    CHECKCUDAERR(cudaMemcpyToSymbol(TRANSCRIPT_GPU_P_ptr, &h_TRANSCRIPT_GPU_P, sizeof(PoseidonBN128GPU::FrElement*))); \
    CHECKCUDAERR(cudaMalloc(&h_TRANSCRIPT_GPU_S, sizeof(PoseidonBN128Constants::S##t_val))); \
    CHECKCUDAERR(cudaMemcpy(h_TRANSCRIPT_GPU_S, PoseidonBN128Constants::S##t_val, sizeof(PoseidonBN128Constants::S##t_val), cudaMemcpyHostToDevice)); \
    CHECKCUDAERR(cudaMemcpyToSymbol(TRANSCRIPT_GPU_S_ptr, &h_TRANSCRIPT_GPU_S, sizeof(PoseidonBN128GPU::FrElement*))); \
} while(0)

void TranscriptBN128_GPU::init_const(uint32_t* gpu_ids, uint32_t num_gpu_ids, uint32_t arity)
{
    if (transcriptBN128_initialized) return;

    int deviceId;
    CHECKCUDAERR(cudaGetDevice(&deviceId));

    // t = arity + 1 (state size + pending size)
    uint32_t t = arity + 1;

    for (uint32_t i = 0; i < num_gpu_ids; i++)
    {
        CHECKCUDAERR(cudaSetDevice(gpu_ids[i]));

        // Initialize constants for the specific t value based on arity
        switch(t) {
            case 2:  INIT_TRANSCRIPT_T_CONSTANTS(2);  break;
            case 3:  INIT_TRANSCRIPT_T_CONSTANTS(3);  break;
            case 4:  INIT_TRANSCRIPT_T_CONSTANTS(4);  break;
            case 5:  INIT_TRANSCRIPT_T_CONSTANTS(5);  break;
            case 6:  INIT_TRANSCRIPT_T_CONSTANTS(6);  break;
            case 7:  INIT_TRANSCRIPT_T_CONSTANTS(7);  break;
            case 8:  INIT_TRANSCRIPT_T_CONSTANTS(8);  break;
            case 9:  INIT_TRANSCRIPT_T_CONSTANTS(9);  break;
            case 10: INIT_TRANSCRIPT_T_CONSTANTS(10); break;
            case 11: INIT_TRANSCRIPT_T_CONSTANTS(11); break;
            case 12: INIT_TRANSCRIPT_T_CONSTANTS(12); break;
            case 13: INIT_TRANSCRIPT_T_CONSTANTS(13); break;
            case 14: INIT_TRANSCRIPT_T_CONSTANTS(14); break;
            case 15: INIT_TRANSCRIPT_T_CONSTANTS(15); break;
            case 16: INIT_TRANSCRIPT_T_CONSTANTS(16); break;
            case 17: INIT_TRANSCRIPT_T_CONSTANTS(17); break;
            default:
                zklog.error("TranscriptBN128_GPU::init_const: Unsupported t value: " + std::to_string(t));
                exitProcess();
                exit(-1);
        }
    }

    CHECKCUDAERR(cudaSetDevice(deviceId));
    transcriptBN128_initialized = 1;
}

#undef INIT_TRANSCRIPT_T_CONSTANTS

void TranscriptBN128_GPU::put(Goldilocks::Element *input, uint64_t size, cudaStream_t stream, TimerGPU *timer)
{
    if(timer != nullptr) TimerStartCategoryGPU((*timer), TRANSCRIPT_PUT);
    _addBN128_GL<<<1, 32, 0, stream>>>(input, size, state, pending, out, pending_cursor, out_cursor, out3_cursor, arity);
    if(timer != nullptr) TimerStopCategoryGPU((*timer), TRANSCRIPT_PUT);
}

void TranscriptBN128_GPU::put(PoseidonBN128GPU::FrElement *input, uint64_t size, cudaStream_t stream, TimerGPU *timer)
{
    if(timer != nullptr) TimerStartCategoryGPU((*timer), TRANSCRIPT_PUT);
    _addBN128_Fr<<<1, 32, 0, stream>>>(input, size, state, pending, out, pending_cursor, out_cursor, out3_cursor, arity);
    if(timer != nullptr) TimerStopCategoryGPU((*timer), TRANSCRIPT_PUT);
}

void TranscriptBN128_GPU::getField(uint64_t *output, cudaStream_t stream)
{
    _getFieldBN128<<<1, 32, 0, stream>>>(output, state, pending, out, out3, pending_cursor, out_cursor, out3_cursor, arity);
}

void TranscriptBN128_GPU::getState(PoseidonBN128GPU::FrElement* output, cudaStream_t stream)
{
    __getStateBN128<<<1, 32, 0, stream>>>(output, state, pending, out, pending_cursor, out_cursor, out3_cursor, arity);
}

void TranscriptBN128_GPU::getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, cudaStream_t stream)
{
    __getPermutationsBN128<<<1, 32, 0, stream>>>(res, n, nBits, state, pending, out, pending_cursor, out_cursor, out3_cursor, arity);
}
