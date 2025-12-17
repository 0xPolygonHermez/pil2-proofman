#ifndef TRANSCRIPT_BN128_CLASS
#define TRANSCRIPT_BN128_CLASS

#include "fr.hpp"
#include "poseidon_opt.hpp"
#include <cstring>
#include "goldilocks_base_field.hpp"

class TranscriptBN128
{
private:
    void _add1(RawFrP::Element input);
    void _updateState();

public:
    uint typeSize = 2;

    uint64_t transcriptArity;

    std::vector<RawFrP::Element> state;
    std::vector<RawFrP::Element> pending;
    std::vector<RawFrP::Element> out;
    std::vector<uint64_t> out3;

    TranscriptBN128(uint64_t arity, bool custom) : state(1, RawFrP::field.zero()), out(1, RawFrP::field.zero()) {
        transcriptArity = custom ? arity : 16;
    }
    
    void put(Goldilocks::Element *input, uint64_t size);
    void put(RawFrP::Element *input, uint64_t size);
    void getState(RawFrP::Element* output);
    void getField(uint64_t *output);

    void getPermutations(uint64_t *res, uint64_t n, uint64_t nBits);
    uint64_t getFields1();
    RawFrP::Element getFields253();
};

#endif