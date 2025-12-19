#include "poseidon_opt.hpp"
#include <omp.h>
#include <cstring>
#include <cassert>

void Poseidon_opt::hash(vector<FrElement> &state, FrElement *result)
{
	hash(state);
	*result = state[0];
}

void Poseidon_opt::hash(vector<FrElement> &state)
{
	assert(state.size() < 18);
	const int t = state.size();
	const int nRoundsP = N_ROUNDS_P[t - 2];

	const vector<FrElement> *c = &(Constants_opt::C[t - 2]);
	const vector<FrElement> *s = &(Constants_opt::S[t - 2]);
	const vector<vector<FrElement>> *m = &(Constants_opt::M[t - 2]);
	const vector<vector<FrElement>> *p = &(Constants_opt::P[t - 2]);

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
		field.add(state[0], state[0], (FrElement &)(*c)[(N_ROUNDS_F / 2 + 1) * t + r]);

		FrElement s0 = field.zero();
		FrElement accumulator1;
		FrElement accumulator2;
		for (int j = 0; j < t; j++)
		{
			accumulator1 = (FrElement &)(*s)[(t * 2 - 1) * r + j];
			field.mul(accumulator1, accumulator1, state[j]);
			field.add(s0, s0, accumulator1);
			if (j > 0)
			{
				accumulator2 = (FrElement &)(*s)[(t * 2 - 1) * r + t + j - 1];
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

void Poseidon_opt::ark(vector<FrElement> *state, const vector<FrElement> *c, const int ssize, int it)
{
	for (int i = 0; i < ssize; i++)
	{
		field.add((*state)[i], (*state)[i], (FrElement &)(*c)[it + i]);
	}
}

void Poseidon_opt::sbox(vector<FrElement> *state, const vector<FrElement> *c, const int ssize, int it)
{
	for (int i = 0; i < ssize; i++)
	{
		exp5((*state)[i]);
		field.add((*state)[i], (*state)[i], (FrElement &)(*c)[it + i]);
	}
}

void Poseidon_opt::exp5(FrElement &r)
{
	FrElement aux = r;
	field.square(r, r);
	field.square(r, r);
	field.mul(r, r, aux);
}

void Poseidon_opt::mix(vector<FrElement> *new_state, vector<FrElement> state, const vector<vector<FrElement>> *m, const int ssize)
{
	for (int i = 0; i < ssize; i++)
	{
		(*new_state)[i] = field.zero();
		for (int j = 0; j < ssize; j++)
		{
			FrElement mji = (*m)[j][i];
			field.mul(mji, mji, state[j]);
			field.add((*new_state)[i], (*new_state)[i], mji);
		}
	}
}

void Poseidon_opt::grinding(uint64_t &nonce, vector<RawFrP::Element> &state, const uint32_t n_bits){
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

			vector<RawFrP::Element> localState(state.size() + 2);
			localState[0] = RawFrP::field.zero();
			std::memcpy(&localState[1], &state[0], state.size() * sizeof(RawFrP::Element));

			// Append nonce
			RawFrP::Element tmp = RawFrP::field.zero();
			tmp.v[0] = offset + i;
			RawFrP::field.toMontgomery(tmp, tmp);
			localState[state.size() + 1] = tmp;

			// Compute hash
			hash(localState);

			RawFrP::Element res;
			RawFrP::field.fromMontgomery(res, localState[0]);

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
