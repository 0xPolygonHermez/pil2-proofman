#ifndef POSEIDON2_BN128_HPP
#define POSEIDON2_BN128_HPP

#include <vector>
#include <string>
#include "bn128.cuh"
#include <cassert>
using namespace std;

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

  void hash(FrElement * d_state, int t);
  
  // Initialize GPU constants (copies all constants to constant memory)
  static void initGPUConstants(uint32_t* gpu_ids, uint32_t num_gpu_ids);
};

__device__ void Poseidon2BN128GPU::pow5(FrElement &x)
{
    FrElement aux;
    field.copy(aux, x);
    field.square(x, x);
    field.square(x, x);
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

#endif // POSEIDON2_BN128_HPP