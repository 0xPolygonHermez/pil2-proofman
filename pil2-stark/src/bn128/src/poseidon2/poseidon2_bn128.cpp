#include "poseidon2_bn128.hpp"

void Poseidon2BN128::hash(vector<FrElement> &state, FrElement *result)
{
	hash(state);
	*result = state[0];
}

void Poseidon2BN128::hash(vector<FrElement> &state)
{

	const int t = state.size();
	assert(t == 2 || t == 3 || t == 4 || t == 8 || t == 12 || t == 16);
	uint32_t pos = t<=4 ? t-2 : t/4 + 1;
	const int nRoundsP = N_ROUNDS_P[pos];
	const FrElement *c = Poseidon2BN128Constants::get_C(t);
	const FrElement *d = Poseidon2BN128Constants::get_D(t);
	matmul_external(&state[0], t);

	
	for (int r = 0; r < N_ROUNDS_F / 2; r++)
	{
		
		pow5add(&state[0], &c[r * t], t);
		matmul_external(&state[0], t);		
	}
	for (int r = 0; r < nRoundsP; r++)
	{
		field.add(state[0], state[0], c[(N_ROUNDS_F / 2) * t + r]);
		pow5(state[0]);
		FrElement sum = field.zero();
		add(sum, &state[0], t);
		prodadd(&state[0], &d[0], sum, t);		
	}
	for (int r = 0; r < N_ROUNDS_F / 2; r++)
	{
		pow5add(&state[0], &c[(N_ROUNDS_F / 2) * t + nRoundsP + r * t], t);
		matmul_external(&state[0], t);
	}
}




