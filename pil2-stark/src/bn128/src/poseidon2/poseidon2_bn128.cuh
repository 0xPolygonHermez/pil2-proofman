#ifndef POSEIDON2_BN128_CUH
#define POSEIDON2_BN128_CUH

#include <vector>
#include <string>
#include "bn128_utils.cuh"
#include "grinding_constants.hpp"
#include <cassert>
using namespace std;

// Full Round count
#define N_ROUNDS_F_POSEIDON2 8

class Poseidon2BN128GPU
{
public:
  typedef BN128GPUScalarField::Element FrElement;
  BN128GPUScalarField field;

  __device__ __forceinline__ void pow5(FrElement &x);
  __device__ __forceinline__ void add(FrElement &x, const FrElement *st, int t);
  __device__ __forceinline__ void prodadd(FrElement *x, const FrElement *D, const FrElement &sum, int t);
  __device__ __forceinline__ void pow5add(FrElement *x, const FrElement *C, int t);
  __device__ __forceinline__ void matmul_m4(FrElement *x);
  __device__ __forceinline__ void matmul_external(FrElement *x, int t);

  __device__ void hash_(FrElement *state, int t, const FrElement *C, const FrElement *D, int nRoundsP);

  __device__ void hash_parallel_(FrElement *shared_state, FrElement *tmp, int t, const FrElement *C, const FrElement *D, int nRoundsP);

  void hash(FrElement *d_state, int t);
  void hashParallel(FrElement *d_state, int t);

  // Initialize GPU constants (copies constants to GPU constant memory)
  static void initGPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
  static void freeGPUConstants();

    // Linear hash for traces stored in row-major layout
    static void linearHash(FrElement *d_output, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int t, cudaStream_t stream);
    // Linear hash for traces stored in tiled layout
    static void linearHashTiles(FrElement *d_output, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, int t, cudaStream_t stream);

    // Merkle tree construction for row-major layout
    static void merkletree(FrElement *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, cudaStream_t stream);
    // Merkle tree construction for tiled layout
    static void merkletreeTiles(FrElement *d_tree, uint64_t *d_input, uint64_t num_cols, uint64_t num_rows, uint64_t arity, cudaStream_t stream);

  // Grinding (proof-of-work nonce search)
  // d_nonceBlock: device buffer for intermediate nonce storage (size: NONCES_LAUNCH_GRID_SIZE * sizeof(uint64_t))
  static void grinding(uint64_t *d_nonce, uint64_t *d_nonceBlock, const FrElement *d_state, uint32_t n_bits, cudaStream_t stream);
};

__device__ void Poseidon2BN128GPU::pow5(FrElement &x)
{
    FrElement aux;
    field.copy(aux, x);
    field.square(x, x); // x^2
    field.square(x, x); // x^4
    field.mul(x, x, aux);
};

__device__ void Poseidon2BN128GPU::add(FrElement &x, const FrElement *st, int t)
{
    for (int i = 0; i < t; i++)
    {
        field.add(x, x, st[i]);
    }
};

__device__ void Poseidon2BN128GPU::prodadd(FrElement *x, const FrElement *D, const FrElement &sum, int t)
{
    for (int i = 0; i < t; i++)
    {
        FrElement tmp;
        field.mul(tmp, x[i], D[i]);
        field.add(x[i], tmp, sum);
    }
};

__device__ void Poseidon2BN128GPU::pow5add(FrElement *x, const FrElement *C, int t)
{
    for (int i = 0; i < t; i++)
    {
        FrElement aux;
        field.add(x[i], x[i], C[i]);
        field.copy(aux, x[i]);
        field.square(x[i], x[i]);
        field.square(x[i], x[i]);
        field.mul(x[i], x[i], aux);
    }
};

__device__ void Poseidon2BN128GPU::matmul_m4(FrElement *x) {
    FrElement t0, t1, t2, t3, t4, t5, t6, t7;
    field.add(t0, x[0], x[1]);
    field.add(t1, x[2], x[3]);
    field.add(t2, x[1], t1);
    field.add(t2, t2, x[1]);
    field.add(t3, x[3], t0);
    field.add(t3, t3, x[3]);
    FrElement t1_2, t0_2;
    field.add(t1_2, t1, t1);
    field.add(t0_2, t0, t0);
    field.add(t4, t1_2, t1_2);
    field.add(t4, t4, t3);
    field.add(t5, t0_2, t0_2);
    field.add(t5, t5, t2);
    field.add(t6, t3, t5);
    field.add(t7, t2, t4);
    
    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
};

__device__ void Poseidon2BN128GPU::matmul_external(FrElement *x, int t) {
    
    switch(t) {
        case 2:
        {
            FrElement sum;
            field.add(sum, x[0], x[1]);
            field.add(x[0], x[0], sum);
            field.add(x[1], x[1], sum);
            return;
        }
        case 3:
        {
            FrElement sum;
            field.add(sum, x[0], x[1]);
            field.add(sum, sum, x[2]);
            field.add(x[0], x[0], sum);
            field.add(x[1], x[1], sum);
            field.add(x[2], x[2], sum);
            return;
        }
        case 4:
        {
            matmul_m4(&x[0]);
            return;
        }
        default:
        {
            for(int i = 0; i < t; i +=4) {
                matmul_m4(&x[i]);
            }   
            FrElement stored[4];
            stored[0] = field.zero();
            stored[1] = field.zero();
            stored[2] = field.zero();
            stored[3] = field.zero();
            for (int i = 0; i < t; i+=4) {
                field.add(stored[0], stored[0], x[i]);
                field.add(stored[1], stored[1], x[i+1]);
                field.add(stored[2], stored[2], x[i+2]);
                field.add(stored[3], stored[3], x[i+3]);
            }
            
            for (int i = 0; i < t; ++i)
            {
                field.add(x[i], x[i], stored[i % 4]);
            };
            return;
        }
    }
    return;
};

// Hash function with constants passed as arguments
__device__ __forceinline__ void Poseidon2BN128GPU::hash_(FrElement *state, int t, const FrElement *C, const FrElement *D, int nRoundsP)
{
    matmul_external(state, t);

    for (int r = 0; r < N_ROUNDS_F_POSEIDON2 / 2; r++) {
        pow5add(state, &C[r * t], t);
        matmul_external(state, t);
    }
    for (int r = 0; r < nRoundsP; r++) {
        BN128GPUScalarField::add(state[0], state[0], C[(N_ROUNDS_F_POSEIDON2 / 2) * t + r]);
        pow5(state[0]);
        FrElement sum = BN128GPUScalarField::zero();
        add(sum, state, t);
        prodadd(state, D, sum, t);
    }
    for (int r = 0; r < N_ROUNDS_F_POSEIDON2 / 2; r++) {
        pow5add(state, &C[(N_ROUNDS_F_POSEIDON2 / 2) * t + nRoundsP + r * t], t);
        matmul_external(state, t);
    }
}

// Parallel hash: each thread handles one state element
// shared_state and tmp must be in shared memory
// Launch with <<<1, 32>>> (single warp)
__device__ __forceinline__ void Poseidon2BN128GPU::hash_parallel_(FrElement *shared_state, FrElement *tmp, int t, const FrElement *C, const FrElement *D, int nRoundsP)
{
    int tid = threadIdx.x;
    bool active = (tid < t);

    // matmul_external (serial, thread 0)
    if (tid == 0) {
        matmul_external(shared_state, t);
    }
    __syncwarp();

    for (int r = 0; r < N_ROUNDS_F_POSEIDON2 / 2; r++) {
        if (active) {
            BN128GPUScalarField::add(shared_state[tid], shared_state[tid], C[r * t + tid]);
            pow5(shared_state[tid]);
        }
        __syncwarp();

        if (tid == 0) {
            matmul_external(shared_state, t);
        }
        __syncwarp();
    }

    for (int r = 0; r < nRoundsP; r++) {
        if (tid == 0) {
            BN128GPUScalarField::add(shared_state[0], shared_state[0], C[(N_ROUNDS_F_POSEIDON2 / 2) * t + r]);
            pow5(shared_state[0]);
        }
        __syncwarp();

        if (tid == 0) {
            FrElement sum = BN128GPUScalarField::zero();
            for (int j = 0; j < t; j++) {
                BN128GPUScalarField::add(sum, sum, shared_state[j]);
            }
            tmp[0] = sum;
        }
        __syncwarp();

        if (active) {
            FrElement s = tmp[0];
            FrElement prod;
            BN128GPUScalarField::mul(prod, shared_state[tid], D[tid]);
            BN128GPUScalarField::add(shared_state[tid], prod, s);
        }
        __syncwarp();
    }

    for (int r = 0; r < N_ROUNDS_F_POSEIDON2 / 2; r++) {
        if (active) {
            BN128GPUScalarField::add(shared_state[tid], shared_state[tid], C[(N_ROUNDS_F_POSEIDON2 / 2) * t + nRoundsP + r * t + tid]);
            pow5(shared_state[tid]);
        }
        __syncwarp();

        if (tid == 0) {
            matmul_external(shared_state, t);
        }
        __syncwarp();
    }
}

#endif // POSEIDON2_BN128_CUH