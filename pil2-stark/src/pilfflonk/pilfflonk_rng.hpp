#ifndef PILFFLONK_RNG_HPP
#define PILFFLONK_RNG_HPP

#include <cstddef>
#include <cstdint>

#include "alt_bn128.hpp"

namespace PilFflonk {

using FrElement = AltBn128::Engine::FrElement;

// Where the prover's blinding factors come from (pilfflonk/docs/protocol.md#blinding): the
// coefficients of each b(X) in p'(X) = p(X) + (X^N − 1)·b(X). The prover draws them in a fixed
// order (Instance::commitStage), so a deterministic source gives the same proof twice.
class BlindingSource {
public:
    virtual ~BlindingSource() = default;

    // Writes n elements of Fr to out, in Montgomery form.
    virtual void fill(FrElement *out, uint64_t n) = 0;
};

// The blinding of a proof (always on), from libsodium:
//
// - random (the default, and the only choice for a real proof): randombytes_buf, the OS's CSPRNG;
// - seeded (insecure, for tests and CI only): a stream that is a function of the 32-byte seed alone.
//   Anyone who knows the seed can remove the blinding, so a seeded proof is not zero-knowledge.
//
// Each element is uniform in Fr: 32 random bytes read little-endian, the top two bits cleared
// (r < 2^254), and drawn again while not below r (probability r/2^254 > 3/4 of acceptance).
//
// The seeded stream is made of blocks of BLOCK_BYTES bytes, block b being
// randombytes_buf_deterministic(BLOCK_BYTES, s_b) with the sub-seed s_b = BLAKE2b-256 (libsodium's
// crypto_generichash) of the 8 bytes of b little-endian, keyed with the seed: every block is an
// independent ChaCha20 stream, so drawing costs the same wherever it is in the stream.
//
// Not safe to use from several threads at once.
class BlindingRng final : public BlindingSource {
public:
    static constexpr size_t SEED_BYTES = 32;
    static constexpr size_t BLOCK_BYTES = 4096;

    // Random. Throws std::runtime_error if libsodium cannot be initialised.
    BlindingRng();

    // Seeded with the SEED_BYTES bytes at seed (insecure, see above). Throws std::runtime_error if
    // libsodium cannot be initialised.
    explicit BlindingRng(const uint8_t *seed);

    bool seeded() const { return isSeeded; }

    void fill(FrElement *out, uint64_t n) override;

private:
    // The next n bytes of the stream (n <= BLOCK_BYTES).
    void bytes(uint8_t *out, size_t n);
    void nextBlock();

    bool isSeeded;
    uint8_t seed[SEED_BYTES] = {};
    uint64_t block = 0;
    uint8_t buffer[BLOCK_BYTES] = {};
    size_t used = BLOCK_BYTES;
};

} // namespace PilFflonk

#endif
