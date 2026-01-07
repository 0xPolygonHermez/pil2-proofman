#include "poseidon2_bn128.cuh"

// Device constant memory definitions
__device__ __constant__ int N_ROUNDS_F = 8;
__device__ __constant__ int N_ROUNDS_P[6] = {56, 56, 56, 57, 57, 57};

void Poseidon2BN128GPU::hash(vector<FrElement> &state, FrElement *result)
{
#if 0
	hash(state);
	*result = state[0];
#endif
}

void Poseidon2BN128GPU::hash(vector<FrElement> &state)
{
	//hash<1,1>(&state[0], nullptr, nullptr, state.size()); //do it better
}

typedef Poseidon2BN128GPU::FrElement FrElementGPU;

__global__ void poseidon2_hash_kernel(FrElementGPU *state, FrElementGPU* C, FrElementGPU* D, int t){

	Poseidon2BN128GPU poseidon;

	//uint32_t pos = t<=4 ? t-2 : t/4 + 1;
	const int nRoundsP = N_ROUNDS_P[0];
	poseidon.matmul_external(&state[0], t);

	for (int r = 0; r < N_ROUNDS_F / 2; r++)
	{		
		poseidon.pow5add(&state[0], &C[r * t], t);
		poseidon.matmul_external(&state[0], t);		
	}
	for (int r = 0; r < nRoundsP; r++)
	{
		BN128GPUScalarField::add(state[0], state[0], C[(N_ROUNDS_F / 2) * t + r]);
		poseidon.pow5(state[0]);
		FrElementGPU sum = BN128GPUScalarField::zero();
		poseidon.add(sum, &state[0], t);
		poseidon.prodadd(&state[0], &D[0], sum, t);		
	}
	for (int r = 0; r < N_ROUNDS_F / 2; r++)
	{
		poseidon.pow5add(&state[0], &C[(N_ROUNDS_F / 2) * t + nRoundsP + r * t], t);
		poseidon.matmul_external(&state[0], t);
	}
 }




