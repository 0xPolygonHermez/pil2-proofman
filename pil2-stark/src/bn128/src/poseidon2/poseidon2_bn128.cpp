#include "poseidon2_bn128.hpp"

void Poseidon2BN128::hash(vector<FrElement> &state, FrElement *result)
{
	hash(state);
	*result = state[0];
}

void Poseidon2BN128::hash(vector<FrElement> &state)
{

	assert(state.size() < 18);
	const int t = state.size();
	assert(t == 4); // Currently only t=4 is supported
	const int nRoundsP = N_ROUNDS_P[t - 1];

	const vector<FrElement> *c = &(Poseidon2BN128Constants::C[t - 1]);
	const vector<FrElement> *d = &(Poseidon2BN128Constants::D[t - 1]);

	matmul_external(&state[0], t);
	for (int r = 0; r < N_ROUNDS_F / 2; r++)
	{
		pow7add(&state[0], &((*c)[r * t]), t);
		matmul_external(&state[0], t);
	}
	for (int r = 0; r < nRoundsP; r++)
	{
		field.add(state[0], state[0], (*c)[(N_ROUNDS_F / 2) * t + r]);
		pow7(state[0]);
		FrElement sum = field.zero();
		add(sum, &state[0], t);
		prodadd(&state[0], &((*d)[0]), sum, t);
	}
	for (int r = 0; r < N_ROUNDS_F / 2; r++)
	{
		pow7add(&state[0], &((*c)[(N_ROUNDS_F / 2) * t + nRoundsP + r * t]), t);
		matmul_external(&state[0], t);
	}
}




