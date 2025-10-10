#ifndef LIB_API_INTERNAL_H
#define LIB_API_INTERNAL_H
#include "starks_api.hpp"

extern ProofDoneCallback proof_done_callback;

struct PackedInfoCPU {
    bool is_packed;
    uint64_t num_packed_words;
    std::vector<uint64_t> unpack_info;
};

struct DeviceCommitBuffersCPU
{
    uint64_t airgroupId;
    uint64_t airId;
    std::string proofType;

    std::map<std::pair<uint64_t, uint64_t>, PackedInfoCPU> packedInfo;

    void addPackedInfoCPU(uint64_t airgroupId, uint64_t airId, uint64_t nCols, bool is_packed, uint64_t num_packed_words, uint64_t* unpack_info_) {
        if (!is_packed) return;
        std::vector<uint64_t> unpack_vec(unpack_info_, unpack_info_ + nCols);
        PackedInfoCPU pInfo = {is_packed, num_packed_words, unpack_vec};
        packedInfo[std::make_pair(airgroupId, airId)] = pInfo;
    }

    PackedInfoCPU* getPackedInfo(uint64_t airgroupId, uint64_t airId) {
        auto it = packedInfo.find({airgroupId, airId});
        if (it != packedInfo.end())
            return &it->second;
        return nullptr;
    }

    void unpack_cpu(
        const uint64_t* src,
        uint64_t* dst,
        uint64_t nRows,
        uint64_t nCols,
        uint64_t words_per_row,
        const std::vector<uint64_t> &unpack_info
    ) {
        // #pragma omp parallel for
        for (uint64_t row = 0; row < nRows; row++) {
            const uint64_t* packed_row = &src[row * words_per_row];
            uint64_t* unpacked_row = &dst[row * nCols];

            uint64_t word = packed_row[0];
            uint64_t word_idx = 0;
            uint64_t bit_offset = 0;

            for (uint64_t c = 0; c < nCols; c++) {
                uint64_t nbits = unpack_info[c];
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
};

#endif