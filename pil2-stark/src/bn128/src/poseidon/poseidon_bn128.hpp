#ifndef POSEIDON_BN128_HPP
#define POSEIDON_BN128_HPP

#include <vector>
#include <string>
#include "ffiasm/fr.hpp"
#include "poseidon_bn128_constants.hpp"
#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include <cassert>
using namespace std;

class PoseidonBN128
{
  typedef RawFr::Element FrElement;

  const static int N_ROUNDS_F = 8;
  const unsigned int N_ROUNDS_P[16] = {56, 57, 56, 60, 60, 63, 64, 63, 60, 66, 60, 65, 70, 60, 64, 68};

private:
  RawFr field;
  void ark(vector<FrElement> *state, const FrElement *c, const int ssize, int it);
  void sbox(vector<FrElement> *state, const FrElement *c, const int ssize, int it);
  void mix(vector<FrElement> *new_state, vector<FrElement> state, const FrElement *m, const int ssize);
  void exp5(FrElement &r);
  void stateExp5(vector<FrElement> *state, const int ssize);

public:
  void hash(vector<FrElement> &state);
  void hash(vector<FrElement> &state, FrElement *result);
  void grinding(uint64_t &nonce, vector<FrElement> &state, const uint32_t n_bits);
  void linearHash(FrElement* output, Goldilocks::Element* input, uint64_t inputSize, uint64_t t, bool custom = false);
  void linearHash(FrElement* output, Goldilocks::Element* trace, uint64_t rows, uint64_t cols, uint64_t t, bool custom = false);
  void merkletree(FrElement* tree, Goldilocks::Element *trace, uint64_t rows, uint64_t cols, uint64_t arity, bool custom = false);

};

#endif // POSEIDON_BN128_HPP