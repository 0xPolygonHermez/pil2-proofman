#include "transcriptGL.hpp"
#include "math.h"

void TranscriptGL::put(Goldilocks::Element *input, uint64_t size)
{
    for (uint64_t i = 0; i < size; i++)
    {
        _add1(input[i]);
    }
}

void TranscriptGL::_updateState() 
{
    while(pending_cursor < transcriptPendingSize) {
        pending[pending_cursor] = Goldilocks::zero();
        pending_cursor++;
    }
    std::memcpy(inputs, pending, transcriptPendingSize * sizeof(Goldilocks::Element));
    std::memcpy(&inputs[transcriptPendingSize], state, transcriptStateSize * sizeof(Goldilocks::Element));
    switch(arity) {
        case 2:
            Poseidon2Goldilocks<8>::hash_full_result_seq(out, inputs);
            break;
        case 3:
            Poseidon2Goldilocks<12>::hash_full_result_seq(out, inputs);
            break;
        case 4:
            Poseidon2Goldilocks<16>::hash_full_result_seq(out, inputs);
            break;
        default:
            zklog.error("TranscriptGL::_updateState: Unsupported arity");
            exitProcess();
            exit(-1);
    }
    out_cursor = transcriptOutSize;
    std::memset(pending, 0, transcriptPendingSize * sizeof(Goldilocks::Element));
    pending_cursor = 0;
    std::memcpy(state, out, transcriptOutSize * sizeof(Goldilocks::Element));
}

void TranscriptGL::_add1(Goldilocks::Element input)
{
    pending[pending_cursor] = input;
    pending_cursor++;
    out_cursor = 0;
    if (pending_cursor == transcriptPendingSize)
    {
        _updateState();
    }
}

void TranscriptGL::getField(uint64_t* output)
{
    for (int i = 0; i < 3; i++)
    {
        Goldilocks::Element val = getFields1();
        output[i] = val.fe;
    }
}

void TranscriptGL::getState(Goldilocks::Element* output) {
    if(pending_cursor > 0) {
        _updateState();
    }
    std::memcpy(output, state, transcriptStateSize * sizeof(Goldilocks::Element));
}

void TranscriptGL::getState(Goldilocks::Element* output, uint64_t nOutputs) {
    if(pending_cursor > 0) {
        _updateState();
    }
    std::memcpy(output, state, nOutputs * sizeof(Goldilocks::Element));
}

Goldilocks::Element TranscriptGL::getFields1()
{
    if (out_cursor == 0)
    {
        _updateState();
    }
    Goldilocks::Element res = out[(transcriptOutSize - out_cursor) % transcriptOutSize];
    out_cursor--;
    return res;
}

void TranscriptGL::getPermutations(uint64_t *res, uint64_t n, uint64_t nBits)
{
    uint64_t totalBits = n * nBits;

    uint64_t NFields = floor((float)(totalBits - 1) / 63) + 1;
    Goldilocks::Element fields[NFields];

    for (uint64_t i = 0; i < NFields; i++)
    {
        fields[i] = getFields1();
    }
    
    uint64_t curField = 0;
    uint64_t curBit = 0;
    for (uint64_t i = 0; i < n; i++)
    {
        uint64_t a = 0;
        for (uint64_t j = 0; j < nBits; j++)
        {
            uint64_t bit = (Goldilocks::toU64(fields[curField]) >> curBit) & 1;
            if (bit)
                a = a + (1 << j);
            curBit++;
            if (curBit == 63)
            {
                curBit = 0;
                curField++;
            }
        }
        res[i] = a;
    }
}