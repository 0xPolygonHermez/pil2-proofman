#include "transcriptGL.hpp"
#include "blake3_goldilocks.hpp"
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
    switch (get_hash_family()) {
    case HashFamily::Poseidon1:
        switch(arity) {
            case 2: PoseidonGoldilocks<8>::permute((Goldilocks::Element(&)[8])*out, (const Goldilocks::Element(&)[8])*inputs, PoseidonMode::Scalar);   break;
            case 3: PoseidonGoldilocks<12>::permute((Goldilocks::Element(&)[12])*out, (const Goldilocks::Element(&)[12])*inputs, PoseidonMode::Scalar); break;
            case 4: PoseidonGoldilocks<16>::permute((Goldilocks::Element(&)[16])*out, (const Goldilocks::Element(&)[16])*inputs, PoseidonMode::Scalar); break;
            default: zklog.error("TranscriptGL::_updateState: Poseidon1 supports arity 2, 3 or 4"); exitProcess(); exit(-1);
        }
        break;
    case HashFamily::Poseidon2:
        switch(arity) {
            case 2: Poseidon2Goldilocks<8>::permute((Goldilocks::Element(&)[8])*out, (const Goldilocks::Element(&)[8])*inputs, Poseidon2Mode::Scalar);   break;
            case 3: Poseidon2Goldilocks<12>::permute((Goldilocks::Element(&)[12])*out, (const Goldilocks::Element(&)[12])*inputs, Poseidon2Mode::Scalar); break;
            case 4: Poseidon2Goldilocks<16>::permute((Goldilocks::Element(&)[16])*out, (const Goldilocks::Element(&)[16])*inputs, Poseidon2Mode::Scalar); break;
            default: zklog.error("TranscriptGL::_updateState: Poseidon2 supports arity 2, 3 or 4"); exitProcess(); exit(-1);
        }
        break;
    case HashFamily::Blake3:
        zklog.error("TranscriptGL::_updateState: unreachable for blake3");
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
    if (get_hash_family() == HashFamily::Blake3)
    {
        const uint64_t w = Goldilocks::toU64(input);
        b3.absorb(&w, 1);
        b3_xof_valid = false;   // the stream changed; old XOF material is stale
        b3_offset = 0;
        b3_ob = 0;
        return;
    }

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
    if (get_hash_family() == HashFamily::Blake3)
    {
        // The BLAKE3 digest so far; does not consume, as the sponge does not.
        uint64_t xof[8];
        b3.finalize_xof(0, xof);
        for (uint32_t i = 0; i < transcriptStateSize; ++i)
            output[i] = Goldilocks::fromU64(xof[i]);
        return;
    }
    if(pending_cursor > 0) {
        _updateState();
    }
    std::memcpy(output, state, transcriptStateSize * sizeof(Goldilocks::Element));
}

void TranscriptGL::getState(Goldilocks::Element* output, uint64_t nOutputs) {
    if (get_hash_family() == HashFamily::Blake3)
    {
        if (nOutputs > 8) {
            zklog.error("TranscriptGL::getState: blake3 yields at most 8 words per XOF block");
            exitProcess();
            exit(-1);
        }
        uint64_t xof[8];
        b3.finalize_xof(0, xof);
        for (uint64_t i = 0; i < nOutputs; ++i)
            output[i] = Goldilocks::fromU64(xof[i]);
        return;
    }
    if(pending_cursor > 0) {
        _updateState();
    }
    std::memcpy(output, state, nOutputs * sizeof(Goldilocks::Element));
}

Goldilocks::Element TranscriptGL::getFields1()
{
    if (get_hash_family() == HashFamily::Blake3)
    {
        // A refill advances only the output-block counter; the root is unchanged.
        if (!b3_xof_valid)
        {
            b3_ob = 0;
            b3.finalize_xof(b3_ob, b3_xof);
            b3_offset = 0;
            b3_xof_valid = true;
        }
        else if (b3_offset == 8)
        {
            ++b3_ob;
            b3.finalize_xof(b3_ob, b3_xof);
            b3_offset = 0;
        }
        return Goldilocks::fromU64(b3_xof[b3_offset++]);
    }

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