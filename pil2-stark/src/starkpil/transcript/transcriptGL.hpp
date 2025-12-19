#ifndef TRANSCRIPT_CLASS
#define TRANSCRIPT_CLASS

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "poseidon2_goldilocks.hpp"
#include "zklog.hpp"



class TranscriptGL
{
private:
    void _add1(Goldilocks::Element input);
    void _updateState();
    Goldilocks::Element getFields1();
    uint32_t arity;

    uint32_t transcriptStateSize;
    uint32_t transcriptPendingSize;
    uint32_t transcriptOutSize;

    Goldilocks::Element *inputs;

public:


    Goldilocks::Element *state;
    Goldilocks::Element *pending;
    Goldilocks::Element *out;

    uint pending_cursor = 0;
    uint out_cursor = 0;
    uint state_cursor = 0;

    TranscriptGL(uint64_t arity, bool custom)
    {
        this->arity = arity;
        transcriptStateSize = HASH_SIZE;
        transcriptPendingSize = 4*(arity - 1);
        transcriptOutSize = 4*arity;

        state = new Goldilocks::Element[transcriptOutSize];
        pending = new Goldilocks::Element[transcriptPendingSize];
        out = new Goldilocks::Element[transcriptOutSize];
        inputs = new Goldilocks::Element[transcriptOutSize];

        std::memset(state, 0, transcriptOutSize * sizeof(Goldilocks::Element));
        std::memset(pending, 0, transcriptPendingSize * sizeof(Goldilocks::Element));
        std::memset(out, 0, transcriptOutSize * sizeof(Goldilocks::Element));
    }
    ~TranscriptGL()
    {
        delete[] state;
        delete[] pending;
        delete[] out;
        delete[] inputs;
    }
    void put(Goldilocks::Element *input, uint64_t size);
    void getField(uint64_t *output);
    void getState(Goldilocks::Element* output);
    void getState(Goldilocks::Element* output, uint64_t nOutputs);
    void getPermutations(uint64_t *res, uint64_t n, uint64_t nBits);
};

#endif