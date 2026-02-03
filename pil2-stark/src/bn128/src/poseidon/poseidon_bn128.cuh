#ifndef POSEIDON_BN128_CUH
#define POSEIDON_BN128_CUH

#include <vector>
#include <string>
#include "bn128.cuh"
#include <cassert>
using namespace std;

// Full Round counts
#define N_ROUNDS_F_POSEIDON 8

class PoseidonBN128GPU
{
public:
    typedef BN128GPUScalarField::Element FrElement;
    BN128GPUScalarField field;

    __device__ __forceinline__ void ark(FrElement *state, const FrElement *c, int t, int offset);
    __device__ __forceinline__ void sbox(FrElement *state, const FrElement *c, int t, int offset);
    __device__ __forceinline__ void mix(FrElement *state, FrElement *tmp, const FrElement *m, int t);
    __device__ __forceinline__ void exp5(FrElement &r);
    __device__ void hash_(FrElement *state, int t, const FrElement *C, const FrElement *M, const FrElement *P, const FrElement *S, int nRoundsP);
    
    // Parallel version: uses blockDim.x threads (should be >= t) to parallelize mix operations
    // Call with <<<1, t>>> or <<<1, 32>>> for warp efficiency
    __device__ void hash_parallel_(FrElement *shared_state, FrElement *tmp, int t, const FrElement *C, const FrElement *M, const FrElement *P, const FrElement *S, int nRoundsP);

    void hash(FrElement *d_state, int t);
    
    // Parallel hash: uses 32 threads (one warp) for parallel operations
    void hashParallel(FrElement *d_state, int t);
    
    // Initialize GPU constants (uploads all t values 2-17)
    static void initGPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
    // Free all GPU constants
    static void freeGPUConstants();
    
    // Linear hash for traces stored in row-major layout (not tiled)
    static void linearHash(FrElement *d_output, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int t, bool custom, cudaStream_t stream);
    
    // Linear hash for traces stored in tiled layout  
    static void linearHashTiles(FrElement *d_output, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int t, bool custom, cudaStream_t stream);
    
    // Merkle tree construction for row-major layout
    static void merkletree(FrElement *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, bool custom, cudaStream_t stream);
    
    // Merkle tree construction for tiled layout
    static void merkletreeTiles(FrElement *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, bool custom, cudaStream_t stream);
    
    // d_nonceBlock: device buffer for intermediate nonce storage (size: gridDim.x * sizeof(uint64_t))
    static void grinding(uint64_t *d_nonce, uint64_t *d_nonceBlock, const FrElement *d_state, uint32_t n_bits, cudaStream_t stream);
};

__device__ void PoseidonBN128GPU::exp5(FrElement &r)
{
    FrElement aux;
    field.copy(aux, r);
    field.square(r, r);
    field.square(r, r);
    field.mul(r, r, aux);
}

__device__ void PoseidonBN128GPU::ark(FrElement *state, const FrElement *c, int t, int offset)
{
    for (int i = 0; i < t; i++)
    {
        field.add(state[i], state[i], c[offset + i]);
    }
}

__device__ void PoseidonBN128GPU::sbox(FrElement *state, const FrElement *c, int t, int offset)
{
    for (int i = 0; i < t; i++)
    {
        exp5(state[i]);
        field.add(state[i], state[i], c[offset + i]);
    }
}

// mix: Matrix multiplication - new_state = M * state
// M is stored in row-major order: M[row*t + col]
// tmp is a pre-allocated temporary buffer of size >= t
__device__ void PoseidonBN128GPU::mix(FrElement *state, FrElement *tmp, const FrElement *m, int t)
{
    for (int i = 0; i < t; i++)
    {
        tmp[i] = BN128GPUScalarField::zero();
        for (int j = 0; j < t; j++)
        {
            FrElement mji;
            field.copy(mji, m[j * t + i]);
            field.mul(mji, mji, state[j]);
            field.add(tmp[i], tmp[i], mji);
        }
    }
    
    for (int i = 0; i < t; i++)
    {
        state[i] = tmp[i];
    }
}

// Hash function with constants passed as arguments 
__device__ __forceinline__ void PoseidonBN128GPU::hash_(FrElement *state, int t, const FrElement *C, const FrElement *M, const FrElement *P, const FrElement *S, int nRoundsP)
{
    PoseidonBN128GPU poseidon;
    
    // Temporary buffer for mix operation
    FrElement tmp[18];
    
    poseidon.ark(state, C, t, 0);
    
    for (int r = 0; r < N_ROUNDS_F_POSEIDON / 2 - 1; r++)
    {
        poseidon.sbox(state, C, t, (r + 1) * t);
        poseidon.mix(state, tmp, M, t);
    }
    
    poseidon.sbox(state, C, t, (N_ROUNDS_F_POSEIDON / 2) * t);
    poseidon.mix(state, tmp, P, t);
    
    for (int r = 0; r < nRoundsP; r++)
    {
        poseidon.exp5(state[0]);
        BN128GPUScalarField::add(state[0], state[0], C[(N_ROUNDS_F_POSEIDON / 2 + 1) * t + r]);

        FrElement s0 = BN128GPUScalarField::zero();
        FrElement accumulator1;
        FrElement accumulator2;
        
        for (int j = 0; j < t; j++)
        {
            accumulator1 = S[(t * 2 - 1) * r + j];
            BN128GPUScalarField::mul(accumulator1, accumulator1, state[j]);
            BN128GPUScalarField::add(s0, s0, accumulator1);
            if (j > 0)
            {
                accumulator2 = S[(t * 2 - 1) * r + t + j - 1];
                BN128GPUScalarField::mul(accumulator2, state[0], accumulator2);
                BN128GPUScalarField::add(state[j], state[j], accumulator2);
            }
        }
        state[0] = s0;
    }
    
    for (int r = 0; r < N_ROUNDS_F_POSEIDON / 2 - 1; r++)
    {
        poseidon.sbox(state, C, t, (N_ROUNDS_F_POSEIDON / 2 + 1) * t + nRoundsP + r * t);
        poseidon.mix(state, tmp, M, t);
    }
    
    for (int i = 0; i < t; i++)
    {
        poseidon.exp5(state[i]);
    }
    poseidon.mix(state, tmp, M, t);
}

// Parallel hash: each thread handles one state element
// shared_state and tmp must be in shared memory
// Launch with <<<1, 32>>> (single warp)
__device__ __forceinline__ void PoseidonBN128GPU::hash_parallel_(FrElement *shared_state, FrElement *tmp, int t, const FrElement *C, const FrElement *M, const FrElement *P, const FrElement *S, int nRoundsP)
{
    int tid = threadIdx.x;
    
    // Only threads 0..t-1 participate in computation
    bool active = (tid < t);
    
    // ark: parallel add
    if (active) {
        BN128GPUScalarField::add(shared_state[tid], shared_state[tid], C[tid]);
    }
    __syncwarp();
    
    // Full rounds)
    for (int r = 0; r < N_ROUNDS_F_POSEIDON / 2 - 1; r++)
    {
        // sbox: parallel exp5 + add
        if (active) {
            exp5(shared_state[tid]);
            BN128GPUScalarField::add(shared_state[tid], shared_state[tid], C[(r + 1) * t + tid]);
        }
        __syncwarp();
        
        // mix: parallel matrix-vector multiply (each thread computes one output)
        if (active) {
            tmp[tid] = BN128GPUScalarField::zero();
            for (int j = 0; j < t; j++) {
                FrElement mji;
                BN128GPUScalarField::copy(mji, M[j * t + tid]);
                BN128GPUScalarField::mul(mji, mji, shared_state[j]);
                BN128GPUScalarField::add(tmp[tid], tmp[tid], mji);
            }
        }
        __syncwarp();
        if (active) {
            shared_state[tid] = tmp[tid];
        }
        __syncwarp();
    }
    
    // sbox before P matrix
    if (active) {
        exp5(shared_state[tid]);
        BN128GPUScalarField::add(shared_state[tid], shared_state[tid], C[(N_ROUNDS_F_POSEIDON / 2) * t + tid]);
    }
    __syncwarp();
    
    // mix with P matrix
    if (active) {
        tmp[tid] = BN128GPUScalarField::zero();
        for (int j = 0; j < t; j++) {
            FrElement pji;
            BN128GPUScalarField::copy(pji, P[j * t + tid]);
            BN128GPUScalarField::mul(pji, pji, shared_state[j]);
            BN128GPUScalarField::add(tmp[tid], tmp[tid], pji);
        }
    }
    __syncwarp();
    if (active) {
        shared_state[tid] = tmp[tid];
    }
    __syncwarp();
    
    // Partial rounds
    for (int r = 0; r < nRoundsP; r++)
    {
        // Only thread 0 does exp5 on state[0]
        if (tid == 0) {
            exp5(shared_state[0]);
            BN128GPUScalarField::add(shared_state[0], shared_state[0], C[(N_ROUNDS_F_POSEIDON / 2 + 1) * t + r]);
        }
        __syncwarp();
        
        // Each thread computes its contribution to s0 and updates state[tid] if tid > 0
        FrElement local_s0 = BN128GPUScalarField::zero();
        if (active) {
            // Contribution to s0: S[r*(2t-1) + tid] * state[tid]
            FrElement s_val;
            BN128GPUScalarField::copy(s_val, S[(t * 2 - 1) * r + tid]);
            BN128GPUScalarField::mul(s_val, s_val, shared_state[tid]);
            local_s0 = s_val;
            
            // For tid > 0: state[tid] += S[r*(2t-1) + t + tid - 1] * state[0]
            if (tid > 0) {
                FrElement s_val2;
                BN128GPUScalarField::copy(s_val2, S[(t * 2 - 1) * r + t + tid - 1]);
                BN128GPUScalarField::mul(s_val2, s_val2, shared_state[0]);
                BN128GPUScalarField::add(shared_state[tid], shared_state[tid], s_val2);
            }
        }
        
        // Store local_s0 in tmp for reduction
        if (active) {
            tmp[tid] = local_s0;
        }
        __syncwarp();
        
        // Thread 0 sums all contributions to get s0
        if (tid == 0) {
            FrElement s0 = BN128GPUScalarField::zero();
            for (int j = 0; j < t; j++) {
                BN128GPUScalarField::add(s0, s0, tmp[j]);
            }
            shared_state[0] = s0;
        }
        __syncwarp();
    }
    
    // Full rounds - second half (minus one)
    for (int r = 0; r < N_ROUNDS_F_POSEIDON / 2 - 1; r++)
    {
        // sbox: parallel exp5 + add
        if (active) {
            exp5(shared_state[tid]);
            BN128GPUScalarField::add(shared_state[tid], shared_state[tid], C[(N_ROUNDS_F_POSEIDON / 2 + 1) * t + nRoundsP + r * t + tid]);
        }
        __syncwarp();
        
        // mix: parallel matrix-vector multiply
        if (active) {
            tmp[tid] = BN128GPUScalarField::zero();
            for (int j = 0; j < t; j++) {
                FrElement mji;
                BN128GPUScalarField::copy(mji, M[j * t + tid]);
                BN128GPUScalarField::mul(mji, mji, shared_state[j]);
                BN128GPUScalarField::add(tmp[tid], tmp[tid], mji);
            }
        }
        __syncwarp();
        if (active) {
            shared_state[tid] = tmp[tid];
        }
        __syncwarp();
    }
    
    // Final exp5 (parallel)
    if (active) {
        exp5(shared_state[tid]);
    }
    __syncwarp();
    
    // Final mix
    if (active) {
        tmp[tid] = BN128GPUScalarField::zero();
        for (int j = 0; j < t; j++) {
            FrElement mji;
            BN128GPUScalarField::copy(mji, M[j * t + tid]);
            BN128GPUScalarField::mul(mji, mji, shared_state[j]);
            BN128GPUScalarField::add(tmp[tid], tmp[tid], mji);
        }
    }
    __syncwarp();
    if (active) {
        shared_state[tid] = tmp[tid];
    }
    __syncwarp();
}

#endif // POSEIDON_BN128_CUH
