#ifndef POSEIDON_BN128_CUH
#define POSEIDON_BN128_CUH

#include <vector>
#include <string>
#include "bn128.cuh"
#include <cassert>
using namespace std;

class PoseidonBN128GPU
{
public:
    typedef BN128GPUScalarField::Element FrElement;
    BN128GPUScalarField field;

    __device__ __forceinline__ void ark(FrElement *state, const FrElement *c, int t, int offset);
    __device__ __forceinline__ void sbox(FrElement *state, const FrElement *c, int t, int offset);
    __device__ __forceinline__ void mix(FrElement *state, FrElement *tmp, const FrElement *m, int t);
    __device__ __forceinline__ void exp5(FrElement &r);

    void hash(FrElement *d_state, int t);
    
    // Must be called once before using GPU hash
    static void initGPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
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

#endif // POSEIDON_BN128_CUH
