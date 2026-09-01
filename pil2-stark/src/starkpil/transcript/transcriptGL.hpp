#ifndef TRANSCRIPT_CLASS
#define TRANSCRIPT_CLASS

#include <memory>

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "starks_api_internal.hpp"
#include "zklog.hpp"

/// One transcript construction. The families share nothing -- a sponge carries its state as block
/// content, blake3 and sha256 carry theirs in a chaining value -- so each gets its own state rather
/// than every instance carrying all three.
class TranscriptImplGL
{
public:
    virtual ~TranscriptImplGL() = default;
    virtual void add1(Goldilocks::Element input) = 0;
    virtual Goldilocks::Element getFields1() = 0;
    /// `nOutputs` words of the state so far. Does not consume.
    virtual void getState(Goldilocks::Element *output, uint64_t nOutputs) = 0;
};

/// Fiat-Shamir transcript, family-agnostic.
///
/// The family is resolved ONCE, in the constructor: `define_hash_family` refuses to change it at
/// runtime, so there is nothing to re-dispatch per call.
class TranscriptGL
{
private:
    std::unique_ptr<TranscriptImplGL> impl;
    uint32_t transcriptStateSize;

public:
    TranscriptGL(uint64_t arity, bool custom);

    void put(Goldilocks::Element *input, uint64_t size);
    void getField(uint64_t *output);
    void getState(Goldilocks::Element *output);
    void getState(Goldilocks::Element *output, uint64_t nOutputs);
    void getPermutations(uint64_t *res, uint64_t n, uint64_t nBits);
};

#endif
