#ifndef TRANSCRIPT_CLASS
#define TRANSCRIPT_CLASS

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "poseidon2_goldilocks.hpp"
#include "zklog.hpp"

#define TRANSCRIPT_STATE_SIZE 4
#define TRANSCRIPT_PENDING_SIZE 8
#define TRANSCRIPT_OUT_SIZE 12

class TranscriptGL
{
private:
    void _add1(Goldilocks::Element input);
    void _updateState();
    Goldilocks::Element getFields1();

public:
    Goldilocks::Element state[TRANSCRIPT_OUT_SIZE];
    Goldilocks::Element pending[TRANSCRIPT_PENDING_SIZE];
    Goldilocks::Element out[TRANSCRIPT_OUT_SIZE];

    uint pending_cursor = 0;
    uint out_cursor = 0;
    uint state_cursor = 0;

    TranscriptGL(uint64_t arity, bool custom)
    {
        std::memset(state, 0, TRANSCRIPT_OUT_SIZE * sizeof(Goldilocks::Element));
        std::memset(pending, 0, TRANSCRIPT_PENDING_SIZE * sizeof(Goldilocks::Element));
        std::memset(out, 0, TRANSCRIPT_OUT_SIZE * sizeof(Goldilocks::Element));
    }
    void put(Goldilocks::Element *input, uint64_t size);
    void getField(uint64_t *output);
    void getState(Goldilocks::Element* output);
    void getState(Goldilocks::Element* output, uint64_t nOutputs);
    void getPermutations(uint64_t *res, uint64_t n, uint64_t nBits);
};

#endif