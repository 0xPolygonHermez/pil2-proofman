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

    void hash(FrElement *d_state, int t);
    
    // Initialize GPU constants (uploads all t values 2-17)
    static void initGPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
    // Free all GPU constants
    static void freeGPUConstants();
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

#endif // POSEIDON_BN128_CUH
