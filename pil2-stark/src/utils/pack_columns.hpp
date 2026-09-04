#ifndef PACK_COLUMNS_HPP
#define PACK_COLUMNS_HPP

#include <cstdint>

// The one copy of the bit-packing format shared by the GPU `unpack` kernel, its CPU twins and the
// packers: each row concatenates its columns (pack_info[c] bits each), padded to words_per_row.

// Bytes of Merkle root at the front of a packed custom-commit file, before words_per_row.
static constexpr uint64_t CUSTOM_COMMIT_ROOT_BYTES = 32;

// Row-major -> packed.
inline void packRowsBits(const uint64_t *src, uint64_t *dst, uint64_t nRows, uint64_t nCols,
                         const uint64_t *pack_info, uint64_t words_per_row)
{
    for (uint64_t row = 0; row < nRows; ++row) {
        uint64_t *packed_row = &dst[row * words_per_row];
        uint64_t word = 0;
        uint64_t bit_offset = 0;
        uint64_t word_idx = 0;

        for (uint64_t c = 0; c < nCols; ++c) {
            uint64_t nbits = pack_info[c];
            uint64_t val = src[row * nCols + c] & ((nbits == 64) ? ~0ULL : ((1ULL << nbits) - 1ULL));
            uint64_t bits_left = 64 - bit_offset;

            if (nbits <= bits_left) {
                word |= (val << bit_offset);
                bit_offset += nbits;
                if (bit_offset == 64) {
                    packed_row[word_idx++] = word;
                    word = 0;
                    bit_offset = 0;
                }
            } else {
                word |= (val << bit_offset);
                packed_row[word_idx++] = word;
                word = val >> bits_left;
                bit_offset = nbits - bits_left;
            }
        }

        if (bit_offset > 0) packed_row[word_idx] = word;
    }
}

// Packed -> row-major. CPU twin of the `unpack` kernel's walk.
inline void unpackRowsBits(const uint64_t *src, uint64_t *dst, uint64_t nRows, uint64_t nCols,
                           const uint64_t *pack_info, uint64_t words_per_row)
{
    for (uint64_t row = 0; row < nRows; ++row) {
        const uint64_t *packed_row = &src[row * words_per_row];
        uint64_t *unpacked_row = &dst[row * nCols];

        uint64_t word = packed_row[0];
        uint64_t word_idx = 0;
        uint64_t bit_offset = 0;

        for (uint64_t c = 0; c < nCols; ++c) {
            uint64_t nbits = pack_info[c];
            uint64_t val;
            uint64_t bits_left = 64 - bit_offset;

            if (nbits <= bits_left) {
                uint64_t mask = (nbits == 64) ? ~0ULL : ((1ULL << nbits) - 1ULL);
                val = (word >> bit_offset) & mask;
                bit_offset += nbits;
                if (bit_offset == 64 && word_idx + 1 < words_per_row) {
                    word = packed_row[++word_idx];
                    bit_offset = 0;
                }
            } else {
                uint64_t low = word >> bit_offset;
                word = packed_row[++word_idx];
                uint64_t high = word & ((1ULL << (nbits - bits_left)) - 1ULL);
                val = (high << bits_left) | low;
                bit_offset = nbits - bits_left;
            }

            unpacked_row[c] = val;
        }
    }
}

// Max bit width of each column of a row-major matrix; returns words_per_row.
inline uint64_t packWidthsRowMajor(const uint64_t *rowMajor, uint64_t nRows, uint64_t nCols,
                                   uint64_t *pack_info)
{
    for (uint64_t c = 0; c < nCols; ++c) pack_info[c] = 1;  // an all-zero column still costs a bit
    for (uint64_t row = 0; row < nRows; ++row) {
        const uint64_t *r = &rowMajor[row * nCols];
        for (uint64_t c = 0; c < nCols; ++c) {
            uint64_t v = r[c];
            uint64_t b = (v == 0) ? 1 : (uint64_t)(64 - __builtin_clzll(v));
            if (b > pack_info[c]) pack_info[c] = b;
        }
    }
    uint64_t total_bits = 0;
    for (uint64_t c = 0; c < nCols; ++c) total_bits += pack_info[c];
    return (total_bits + 63) / 64;
}

#endif
