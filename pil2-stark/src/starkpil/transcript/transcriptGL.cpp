#include "transcriptGL.hpp"
#include "blake3_core.hpp"
#include "blake3_goldilocks.hpp"
#include "sha256_core.hpp"
#include "sha256_goldilocks.hpp"
#include "poseidon_goldilocks.hpp"
#include "poseidon2_goldilocks.hpp"
#include "math.h"

#include <cstring>

namespace {

// ── sponge (Poseidon1 / Poseidon2) ───────────────────────────────────────────
class SpongeTrxImpl : public TranscriptImplGL
{
public:
    explicit SpongeTrxImpl(uint64_t arity_) : arity((uint32_t)arity_)
    {
        transcriptPendingSize = 4 * (arity - 1);
        transcriptOutSize = 4 * arity;
        state = new Goldilocks::Element[transcriptOutSize];
        pending = new Goldilocks::Element[transcriptPendingSize];
        out = new Goldilocks::Element[transcriptOutSize];
        inputs = new Goldilocks::Element[transcriptOutSize];
        std::memset(state, 0, transcriptOutSize * sizeof(Goldilocks::Element));
        std::memset(pending, 0, transcriptPendingSize * sizeof(Goldilocks::Element));
        std::memset(out, 0, transcriptOutSize * sizeof(Goldilocks::Element));
    }
    ~SpongeTrxImpl() override
    {
        delete[] state;
        delete[] pending;
        delete[] out;
        delete[] inputs;
    }

    void add1(Goldilocks::Element input) override
    {
        pending[pending_cursor++] = input;
        out_cursor = 0;
        if (pending_cursor == transcriptPendingSize) updateState();
    }

    Goldilocks::Element getFields1() override
    {
        if (out_cursor == 0) updateState();
        Goldilocks::Element res = out[(transcriptOutSize - out_cursor) % transcriptOutSize];
        out_cursor--;
        return res;
    }

    void getState(Goldilocks::Element *output, uint64_t nOutputs) override
    {
        if (pending_cursor > 0) updateState();
        std::memcpy(output, state, nOutputs * sizeof(Goldilocks::Element));
    }

private:
    void updateState()
    {
        while (pending_cursor < transcriptPendingSize) pending[pending_cursor++] = Goldilocks::zero();
        std::memcpy(inputs, pending, transcriptPendingSize * sizeof(Goldilocks::Element));
        std::memcpy(&inputs[transcriptPendingSize], state, HASH_SIZE * sizeof(Goldilocks::Element));

        switch (get_hash_family()) {
        case HashFamily::Poseidon1:
            switch (arity) {
                case 2: PoseidonGoldilocks<8>::permute((Goldilocks::Element(&)[8])*out, (const Goldilocks::Element(&)[8])*inputs, PoseidonMode::Scalar);   break;
                case 3: PoseidonGoldilocks<12>::permute((Goldilocks::Element(&)[12])*out, (const Goldilocks::Element(&)[12])*inputs, PoseidonMode::Scalar); break;
                case 4: PoseidonGoldilocks<16>::permute((Goldilocks::Element(&)[16])*out, (const Goldilocks::Element(&)[16])*inputs, PoseidonMode::Scalar); break;
                default: zklog.error("SpongeTrxImpl: Poseidon1 supports arity 2, 3 or 4"); exitProcess(); exit(-1);
            }
            break;
        case HashFamily::Poseidon2:
            switch (arity) {
                case 2: Poseidon2Goldilocks<8>::permute((Goldilocks::Element(&)[8])*out, (const Goldilocks::Element(&)[8])*inputs, Poseidon2Mode::Scalar);   break;
                case 3: Poseidon2Goldilocks<12>::permute((Goldilocks::Element(&)[12])*out, (const Goldilocks::Element(&)[12])*inputs, Poseidon2Mode::Scalar); break;
                case 4: Poseidon2Goldilocks<16>::permute((Goldilocks::Element(&)[16])*out, (const Goldilocks::Element(&)[16])*inputs, Poseidon2Mode::Scalar); break;
                default: zklog.error("SpongeTrxImpl: Poseidon2 supports arity 2, 3 or 4"); exitProcess(); exit(-1);
            }
            break;
        default:
            zklog.error("SpongeTrxImpl: reached with a non-sponge family");
            exitProcess();
            exit(-1);
        }

        out_cursor = transcriptOutSize;
        std::memset(pending, 0, transcriptPendingSize * sizeof(Goldilocks::Element));
        pending_cursor = 0;
        std::memcpy(state, out, transcriptOutSize * sizeof(Goldilocks::Element));
    }

    uint32_t arity;
    uint32_t transcriptPendingSize;
    uint32_t transcriptOutSize;
    Goldilocks::Element *state;
    Goldilocks::Element *pending;
    Goldilocks::Element *out;
    Goldilocks::Element *inputs;
    uint32_t pending_cursor = 0;
    uint32_t out_cursor = 0;
};

// ── blake3: the stream is blake3(canonical-LE words), challenges come from its XOF ───
class Blake3TrxImpl : public TranscriptImplGL
{
public:
    Blake3TrxImpl() { h.init(); }

    void add1(Goldilocks::Element input) override
    {
        const uint64_t w = Goldilocks::toU64(input);
        h.absorb(&w, 1);
        valid = false;   // the stream changed; old XOF material is stale
        offset = 0;
        ob = 0;
    }

    Goldilocks::Element getFields1() override
    {
        // A refill advances only the output-block counter; the root is unchanged.
        if (!valid) { ob = 0; h.finalize_xof(ob, xof); offset = 0; valid = true; }
        else if (offset == 8) { ++ob; h.finalize_xof(ob, xof); offset = 0; }
        return Goldilocks::fromU64(xof[offset++]);
    }

    void getState(Goldilocks::Element *output, uint64_t nOutputs) override
    {
        if (nOutputs > 8) {
            zklog.error("Blake3TrxImpl: blake3 yields at most 8 words per XOF block");
            exitProcess();
            exit(-1);
        }
        uint64_t block[8];
        h.finalize_xof(0, block);   // digest so far; does NOT consume
        for (uint64_t i = 0; i < nOutputs; ++i) output[i] = Goldilocks::fromU64(block[i]);
    }

private:
    blake3core::Hasher h;
    uint64_t xof[8];
    uint32_t offset = 0;   // words consumed from xof
    uint32_t ob = 0;       // which XOF output block xof holds
    bool valid = false;
};

// ── sha256: same shape, but no XOF -- a squeeze is SHA256(digest || ctr), 4 words ───
class Sha256TrxImpl : public TranscriptImplGL
{
public:
    Sha256TrxImpl() { h.init(); }

    void add1(Goldilocks::Element input) override
    {
        const uint64_t w = Goldilocks::toU64(input);
        h.absorb(&w, 1);
        valid = false;
        offset = 0;
        ctr = 0;
    }

    Goldilocks::Element getFields1() override
    {
        if (!valid) { ctr = 0; h.squeeze(ctr, sq); offset = 0; valid = true; }
        else if (offset == 4) { ++ctr; h.squeeze(ctr, sq); offset = 0; }
        return Goldilocks::fromU64(sq[offset++]);
    }

    void getState(Goldilocks::Element *output, uint64_t nOutputs) override
    {
        // Capped at 8 to match blake3's contract; a squeeze is 4 words, so 8 is two counters.
        if (nOutputs > 8) {
            zklog.error("Sha256TrxImpl: sha256 yields at most 8 words (two squeezes)");
            exitProcess();
            exit(-1);
        }
        uint64_t block[4];
        for (uint64_t i = 0; i < nOutputs; ++i) {
            if (i % 4 == 0) h.squeeze(i / 4, block);   // does NOT consume
            output[i] = Goldilocks::fromU64(block[i % 4]);
        }
    }

private:
    sha256core::Hasher h;
    uint64_t sq[4];
    uint32_t offset = 0;   // words consumed from sq
    uint64_t ctr = 0;      // which squeeze block sq holds
    bool valid = false;
};

std::unique_ptr<TranscriptImplGL> makeImpl(uint64_t arity)
{
    switch (get_hash_family()) {
    case HashFamily::Blake3: return std::make_unique<Blake3TrxImpl>();
    case HashFamily::Sha256: return std::make_unique<Sha256TrxImpl>();
    default:                 return std::make_unique<SpongeTrxImpl>(arity);
    }
}

}  // namespace

TranscriptGL::TranscriptGL(uint64_t arity, bool custom) : impl(makeImpl(arity)), transcriptStateSize(HASH_SIZE)
{
    (void)custom;
}

void TranscriptGL::put(Goldilocks::Element *input, uint64_t size)
{
    for (uint64_t i = 0; i < size; i++) impl->add1(input[i]);
}

void TranscriptGL::getField(uint64_t *output)
{
    // RAW .fe, not toU64: a sponge permutation's output is not guaranteed canonical, and reducing
    // it here would change the challenges for the poseidon families.
    for (int i = 0; i < 3; i++) output[i] = impl->getFields1().fe;
}

void TranscriptGL::getState(Goldilocks::Element *output)
{
    impl->getState(output, transcriptStateSize);
}

void TranscriptGL::getState(Goldilocks::Element *output, uint64_t nOutputs)
{
    impl->getState(output, nOutputs);
}

void TranscriptGL::getPermutations(uint64_t *res, uint64_t n, uint64_t nBits)
{
    uint64_t totalBits = n * nBits;
    uint64_t NFields = floor((float)(totalBits - 1) / 63) + 1;
    Goldilocks::Element fields[NFields];

    for (uint64_t i = 0; i < NFields; i++) fields[i] = impl->getFields1();

    uint64_t curField = 0;
    uint64_t curBit = 0;
    for (uint64_t i = 0; i < n; i++)
    {
        uint64_t a = 0;
        for (uint64_t j = 0; j < nBits; j++)
        {
            uint64_t bit = (Goldilocks::toU64(fields[curField]) >> curBit) & 1;
            if (bit) a = a + (1 << j);
            curBit++;
            if (curBit == 63) { curBit = 0; curField++; }
        }
        res[i] = a;
    }
}
