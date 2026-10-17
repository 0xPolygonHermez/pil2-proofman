#include "pilfflonk_rng.hpp"

#include <sodium.h>

#include <cstring>
#include <stdexcept>

#include "pilfflonk_fr.hpp"
#include "pilfflonk_transcript.hpp"

namespace PilFflonk {

namespace {

static_assert(BlindingRng::SEED_BYTES == randombytes_SEEDBYTES, "the seed is randombytes_buf_deterministic's");
static_assert(BlindingRng::SEED_BYTES == crypto_generichash_BYTES, "a sub-seed is a BLAKE2b-256 hash");
static_assert(BlindingRng::SEED_BYTES >= crypto_generichash_KEYBYTES_MIN &&
                  BlindingRng::SEED_BYTES <= crypto_generichash_KEYBYTES_MAX,
              "the seed keys BLAKE2b");
static_assert(BlindingRng::BLOCK_BYTES % FR_BYTES == 0, "a candidate never straddles two blocks");

void initSodium() {
    // Thread-safe, and a no-op after the first call.
    if (sodium_init() < 0) {
        throw std::runtime_error("BlindingRng: libsodium cannot be initialised");
    }
}

} // namespace

BlindingRng::BlindingRng() : isSeeded(false) {
    initSodium();
}

BlindingRng::BlindingRng(const uint8_t *_seed) : isSeeded(true) {
    initSodium();
    std::memcpy(seed, _seed, SEED_BYTES);
}

void BlindingRng::nextBlock() {
    uint8_t index[sizeof(block)];
    for (size_t i = 0; i < sizeof(block); ++i) {
        index[i] = static_cast<uint8_t>(block >> (8 * i));
    }
    uint8_t subSeed[SEED_BYTES];
    if (crypto_generichash(subSeed, sizeof(subSeed), index, sizeof(index), seed, SEED_BYTES) != 0) {
        throw std::runtime_error("BlindingRng: BLAKE2b failed");
    }
    randombytes_buf_deterministic(buffer, BLOCK_BYTES, subSeed);
    sodium_memzero(subSeed, sizeof(subSeed));
    ++block;
    used = 0;
}

void BlindingRng::bytes(uint8_t *out, size_t n) {
    if (!isSeeded) {
        randombytes_buf(out, n);
        return;
    }
    if (used + n > BLOCK_BYTES) {
        nextBlock();
    }
    std::memcpy(out, buffer + used, n);
    used += n;
}

void BlindingRng::fill(FrElement *out, uint64_t n) {
    uint8_t candidate[FR_BYTES];
    for (uint64_t i = 0; i < n; ++i) {
        do {
            bytes(candidate, sizeof(candidate));
            candidate[FR_BYTES - 1] &= 0x3f;
        } while (decodeFr(candidate, out[i]) != AbsorbError::None);
    }
    sodium_memzero(candidate, sizeof(candidate));
}

} // namespace PilFflonk
