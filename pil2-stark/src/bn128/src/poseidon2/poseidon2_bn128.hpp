#ifndef POSEIDON2_BN128_HPP
#define POSEIDON2_BN128_HPP

#include <vector>
#include <string>
#include "fr.hpp"
#include "poseidon2_bn128_constants.hpp"
#include <cassert>
using namespace std;

class Poseidon2BN128
{
  typedef RawFr::Element FrElement;

  const static int N_ROUNDS_F = 8;
  const unsigned int N_ROUNDS_P[6] = {56, 56, 56, 57, 57, 57}; //for t=2,3,4,8,12,16

private:
  RawFr field;

  inline void pow5(FrElement &x);
  inline void add(FrElement &x, const FrElement *st, int t);
  inline void prodadd(FrElement *x, const FrElement *D, const FrElement &sum, int t);
  inline void pow5add(FrElement *x, const FrElement *C, int t);
  inline void matmul_m4(FrElement *x);
  inline void matmul_external(FrElement *x, int t);


public:
  void hash(vector<FrElement> &state);
  void hash(vector<FrElement> &state, FrElement *result);
};

void Poseidon2BN128::pow5(FrElement &x)
{
    FrElement aux;
    field.copy(aux, x);
    field.square(x, x);
    field.square(x, x);
    field.mul(x, x, aux);
};

void Poseidon2BN128::add(FrElement &x, const FrElement *st, int t)
{
    for (int i = 0; i < t; i++)
    {
        field.add(x, x, st[i]);
    }
};

void Poseidon2BN128::prodadd(FrElement *x, const FrElement *D, const FrElement &sum, int t)
{
    for (int i = 0; i < t; i++)
    {
        FrElement tmp;
        field.mul(tmp, x[i], D[i]);
        field.add(x[i], tmp, sum);
    }
};

void Poseidon2BN128::pow5add(FrElement *x, const FrElement *C, int t)
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

void Poseidon2BN128::matmul_m4(FrElement *x) {
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

void Poseidon2BN128::matmul_external(FrElement *x, int t) {
    
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