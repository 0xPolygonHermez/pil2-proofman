#include "poseidon2_bn128.hpp"
#include <omp.h>
#include <cstring>

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

void Poseidon2BN128::linearHash(FrElement* output, Goldilocks::Element* input, uint64_t inputSize, uint64_t t)
{
	FrElement result = field.zero();

	uint64_t nElementsGL = (inputSize + 2) / 3;
	RawFr::Element* buff = (RawFr::Element *)malloc(nElementsGL * sizeof(RawFr::Element));

	for (uint64_t j = 0; j < nElementsGL; j++)
	{
		buff[j] = RawFr::field.zero();
		uint64_t pending = inputSize - j * 3;
		uint64_t batch;
		(pending >= 3) ? batch = 3 : batch = pending;
		for (uint64_t k = 0; k < batch; k++)
		{
			buff[j].v[k] = Goldilocks::toU64(input[j * 3 + k]);
		}
		RawFr::field.toMontgomery(buff[j], buff[j]);
	}

	uint64_t pending = nElementsGL;
	std::vector<RawFr::Element> elements(t);
	while (pending > 0)
	{
		std::memset(&elements[0], 0, t * sizeof(RawFr::Element));
		uint64_t batch = (pending >= (uint64_t)(t-1)) ? (uint64_t)(t-1) : pending;
		std::memcpy(&elements[1], &buff[nElementsGL - pending], batch * sizeof(RawFr::Element));
		elements[0] = result;
		hash(elements, &result);
		pending -= batch;
	}
	free(buff);

	*output = result;
}

void Poseidon2BN128::linearHash(FrElement* output, Goldilocks::Element* trace, uint64_t rows, uint64_t cols, uint64_t t)
{
	#pragma omp parallel for
	for (uint64_t i = 0; i < rows; i++)
	{
		linearHash(&output[i], &trace[i * cols], cols, t);
	}
}

void Poseidon2BN128::merkletree(FrElement* tree, Goldilocks::Element *trace, uint64_t rows, uint64_t cols, uint64_t arity)
{
	
	linearHash(tree, trace, rows, cols, arity);

	//Build tree levels (compressor mode: no capacity)
	RawFr::Element *cursor = &tree[0];
	uint64_t n = rows;
	uint64_t nextN = (n + arity - 1) / arity;
	RawFr::Element *cursorNext = &cursor[nextN * arity];

	while (n > 1)
	{
		uint64_t batches = (n + arity - 1) / arity;
#pragma omp parallel for
		for (uint64_t i = 0; i < batches; i++)
		{
			vector<RawFr::Element> elements(arity);
			std::memset(&elements[0], 0, arity * sizeof(RawFr::Element));
			uint64_t numElements = (i == batches - 1) ? n - i*arity : arity;
			std::memcpy(&elements[0], &cursor[i * arity], numElements * sizeof(RawFr::Element));
			hash(elements, &cursorNext[i]);
		}

		n = nextN;
		nextN = (n + arity - 1) / arity;
		cursor = cursorNext;
		cursorNext = &cursor[nextN * arity];
	}
}

void Poseidon2BN128::grinding(uint64_t &nonce, vector<RawFr::Element> &state, const uint32_t n_bits)
{
	uint64_t checkChunk = omp_get_max_threads() * 512;
	uint64_t level = uint64_t(1) << (64 - n_bits);
	uint64_t* chunkIdxs = new uint64_t[omp_get_max_threads()];
	uint64_t offset = 0;
	nonce = UINT64_MAX;

	for (int i = 0; i < omp_get_max_threads(); ++i)
		chunkIdxs[i] = UINT64_MAX;

	uint64_t max_k = 1ULL << n_bits;

	for (uint64_t k = 0; k < max_k; ++k) {
		#pragma omp parallel for
		for (uint64_t i = 0; i < checkChunk; i++) {
			int tid = omp_get_thread_num();

			if (chunkIdxs[tid] != UINT64_MAX) continue;

			// localState = [state[0], state[1], state[2], nonce]
			vector<RawFr::Element> localState(state.size() + 1);
			std::memcpy(&localState[0], &state[0], state.size() * sizeof(RawFr::Element));

			// Append nonce
			RawFr::Element tmp = RawFr::field.zero();
			tmp.v[0] = offset + i;
			RawFr::field.toMontgomery(tmp, tmp);
			localState[state.size()] = tmp;

			// Compute hash
			hash(localState);

			RawFr::Element res;
			RawFr::field.fromMontgomery(res, localState[0]);

			if (res.v[0] < level) {
				chunkIdxs[tid] = offset + i;
			}
		}

		// Collect the first found nonce
		for (int i = 0; i < omp_get_max_threads(); ++i) {
			if (chunkIdxs[i] != UINT64_MAX) {
				nonce = chunkIdxs[i];
				break;
			}
		}

		if (nonce != UINT64_MAX)
			break;

		offset += checkChunk;
	}

	if (nonce == UINT64_MAX)
		throw std::runtime_error("Poseidon2::grinding: could not find a valid nonce");

	delete[] chunkIdxs;
}




