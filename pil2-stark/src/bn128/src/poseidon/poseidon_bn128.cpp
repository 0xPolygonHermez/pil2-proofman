#include "poseidon_bn128.hpp"
#include <omp.h>
#include <cstring>
//#include <cassert>

void PoseidonBN128::hash(vector<FrElement> &state, FrElement *result)
{
	hash(state);
	*result = state[0];
}

void PoseidonBN128::hash(vector<FrElement> &state)
{
	assert(state.size() < 18);
	const int t = state.size();
	const int nRoundsP = N_ROUNDS_P[t - 2];

	const FrElement *c = (const FrElement *)PoseidonBN128Constants::get_C(t);
	const FrElement *s = (const FrElement *)PoseidonBN128Constants::get_S(t);
	const FrElement *m = (const FrElement *)PoseidonBN128Constants::get_M(t);
	const FrElement *p = (const FrElement *)PoseidonBN128Constants::get_P(t);

	ark(&state, c, t, 0);
	for (int r = 0; r < N_ROUNDS_F / 2 - 1; r++)
	{
		sbox(&state, c, t, (r + 1) * t);
		mix(&state, state, m, t);
	}
	sbox(&state, c, t, (N_ROUNDS_F / 2 - 1 + 1) * t);
	mix(&state, state, p, t);
	for (int r = 0; r < nRoundsP; r++)
	{
		exp5(state[0]);
		field.add(state[0], state[0], (FrElement &)c[(N_ROUNDS_F / 2 + 1) * t + r]);
		FrElement s0 = field.zero();
		FrElement accumulator1;
		FrElement accumulator2;
		for (int j = 0; j < t; j++)
		{
			accumulator1 = (FrElement &)s[(t * 2 - 1) * r + j];
			field.mul(accumulator1, accumulator1, state[j]);
			field.add(s0, s0, accumulator1);
			if (j > 0)
			{
				accumulator2 = (FrElement &)s[(t * 2 - 1) * r + t + j - 1];
				field.mul(accumulator2, state[0], accumulator2);
				field.add(state[j], state[j], accumulator2);
			}
		}
		state[0] = s0;
	}
	for (int r = 0; r < N_ROUNDS_F / 2 - 1; r++)
	{
		sbox(&state, c, t, (N_ROUNDS_F / 2 + 1) * t + nRoundsP + r * t);
		mix(&state, state, m, t);
	}
	for (int i = 0; i < t; i++)
	{
		exp5(state[i]);
	}
	mix(&state, state, m, t);
}

void PoseidonBN128::ark(vector<FrElement> *state, const FrElement *c, const int ssize, int it)
{
	for (int i = 0; i < ssize; i++)
	{
		field.add((*state)[i], (*state)[i], (FrElement &)c[it + i]);
	}
}

void PoseidonBN128::sbox(vector<FrElement> *state, const FrElement *c, const int ssize, int it)
{
	for (int i = 0; i < ssize; i++)
	{
		exp5((*state)[i]);
		field.add((*state)[i], (*state)[i], (FrElement &)c[it + i]);
	}
}

void PoseidonBN128::exp5(FrElement &r)
{
	FrElement aux = r;
	field.square(r, r);
	field.square(r, r);
	field.mul(r, r, aux);
}

void PoseidonBN128::mix(vector<FrElement> *new_state, vector<FrElement> state, const FrElement *m, const int ssize)
{
	for (int i = 0; i < ssize; i++)
	{
		(*new_state)[i] = field.zero();
		for (int j = 0; j < ssize; j++)
		{
			FrElement mji = (FrElement &)m[j * ssize + i];
			field.mul(mji, mji, state[j]);
			field.add((*new_state)[i], (*new_state)[i], mji);
		}
	}
}

void PoseidonBN128::grinding(uint64_t &nonce, vector<RawFr::Element> &state, const uint32_t n_bits){
    uint64_t checkChunk = omp_get_max_threads() * 512;
	uint64_t level   = uint64_t(1) << (64 - n_bits);
    uint64_t* chunkIdxs = new uint64_t[omp_get_max_threads()];
    uint64_t offset = 0;
    nonce = UINT64_MAX;

    for(int i = 0; i < omp_get_max_threads(); ++i)
        chunkIdxs[i] = UINT64_MAX;

    uint64_t max_k = 1ULL << n_bits;

    for(uint64_t k = 0; k < max_k; ++k){
		#pragma omp parallel for
		for (uint64_t i = 0; i < checkChunk; i++) {
			int tid = omp_get_thread_num();

			if (chunkIdxs[tid] != UINT64_MAX) continue;

			vector<RawFr::Element> localState(state.size() + 2);
			localState[0] = RawFr::field.zero();
			std::memcpy(&localState[1], &state[0], state.size() * sizeof(RawFr::Element));

			// Append nonce
			RawFr::Element tmp = RawFr::field.zero();
			tmp.v[0] = offset + i;
			RawFr::field.toMontgomery(tmp, tmp);
			localState[state.size() + 1] = tmp;

			// Compute hash
			hash(localState);

			RawFr::Element res;
			RawFr::field.fromMontgomery(res, localState[0]);

			if(res.v[0] < level) {
				chunkIdxs[tid] = offset + i;
			}
		}
        

        // Collect the first found nonce
        for(int i = 0; i < omp_get_max_threads(); ++i){
            if (chunkIdxs[i] != UINT64_MAX){
                nonce = chunkIdxs[i];
                break;
            }
        }

        if (nonce != UINT64_MAX)
            break;

        offset += checkChunk;
    }

    if(nonce == UINT64_MAX)
        throw std::runtime_error("Poseidon_opt::grinding: could not find a valid nonce");

    delete[] chunkIdxs;
}
