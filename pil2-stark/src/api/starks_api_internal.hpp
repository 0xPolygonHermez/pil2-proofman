#ifndef LIB_API_INTERNAL_H
#define LIB_API_INTERNAL_H
#include "starks_api.hpp"
#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include <utility>


#include "hash_family.hpp"
#include "poseidon_goldilocks.hpp"
#include "poseidon2_goldilocks.hpp"
#include "blake3_goldilocks.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

inline void runGrinding(uint64_t &nonce,
                        const uint64_t *challenge, uint32_t powBits) {
    switch (get_hash_family()) {
        case HashFamily::Poseidon1: PoseidonGoldilocks<8>::grinding(nonce, challenge, powBits); break;
        case HashFamily::Poseidon2: Poseidon2GoldilocksGrinding::grinding(nonce, challenge, powBits); break;
        case HashFamily::Blake3:    Blake3Goldilocks::grinding(nonce, challenge, powBits); break;
        default: break;
    }
}

inline void runGrindingPermute(Goldilocks::Element (&out)[8],
                               const Goldilocks::Element (&in)[8]) {
    switch (get_hash_family()) {
        case HashFamily::Poseidon1: PoseidonGoldilocks<8>::permute(out, in, PoseidonMode::Scalar);   break;
        case HashFamily::Poseidon2: Poseidon2Goldilocks<8>::permute(out, in, Poseidon2Mode::Scalar); break;
        case HashFamily::Blake3:    Blake3Goldilocks::permute(out, in); break;
        default: break;
    }
}

extern ProofDoneCallback proof_done_callback;

struct PackedInfoCPU {
    bool is_packed;
    uint64_t num_packed_words;
    std::vector<uint64_t> unpack_info;
    // Indexed variant descriptor (empty col_source when the air is not indexed).
    std::vector<uint8_t> col_source; // per column: 0 = row stream, 1 = table stream
    uint64_t index_bits = 0;
    uint64_t words_per_entry = 0;
    bool indexed() const { return !col_source.empty(); }
};

// Read `nbits` from a packed stream at cursor (word,idx,off), advancing it.
// Mirrors the GPU idx_read_bits bit-walk exactly.
static inline uint64_t cpu_idx_read_bits(
    const uint64_t* base, uint64_t words, uint64_t &word, uint64_t &idx, uint64_t &off, uint64_t nbits)
{
    uint64_t val;
    uint64_t bits_left = 64 - off;
    if (nbits <= bits_left) {
        uint64_t mask = (nbits == 64) ? ~0ULL : ((1ULL << nbits) - 1ULL);
        val = (word >> off) & mask;
        off += nbits;
        if (off == 64 && idx + 1 < words) { word = base[++idx]; off = 0; }
    } else {
        uint64_t low = word >> off;
        word = base[++idx];
        uint64_t high = word & ((1ULL << (nbits - bits_left)) - 1ULL);
        val = (high << bits_left) | low;
        off = nbits - bits_left;
    }
    return val;
}

struct DeviceCommitBuffersCPU
{
    uint64_t airgroupId;
    uint64_t airId;
    std::string proofType;

    bool packedTrace = false;

    std::map<std::pair<uint64_t, uint64_t>, PackedInfoCPU> packedInfo;
    // Per-program instruction tables for indexed airs (num_entries * words_per_entry words).
    std::map<std::pair<uint64_t, uint64_t>, std::vector<uint64_t>> instrTables;
    std::map<std::pair<uint64_t, uint64_t>, uint64_t> instrEntries; // entry count, for index bounds

    void addPackedInfoCPU(uint64_t airgroupId, uint64_t airId, uint64_t nCols, bool is_packed,
                          uint64_t num_packed_words, uint64_t* unpack_info_, uint8_t* col_source_,
                          uint64_t index_bits, uint64_t words_per_entry) {
        if (!is_packed) return;
        std::vector<uint64_t> unpack_vec(unpack_info_, unpack_info_ + nCols);
        std::vector<uint8_t> col_source_vec;
        if (col_source_ != nullptr) col_source_vec.assign(col_source_, col_source_ + nCols);
        PackedInfoCPU pInfo = {is_packed, num_packed_words, unpack_vec, col_source_vec, index_bits, words_per_entry};
        packedInfo[std::make_pair(airgroupId, airId)] = pInfo;
    }

    void registerInstructionTable(uint64_t airgroupId, uint64_t airId, const uint64_t* table, uint64_t num_entries, uint64_t words_per_entry) {
        auto key = std::make_pair(airgroupId, airId);
        instrTables[key].assign(table, table + num_entries * words_per_entry);
        instrEntries[key] = num_entries;
        // words_per_entry is also carried in PackedInfoCPU (from setup). Keep them in
        // step: register_instruction_table is the authority for the live program.
        auto pit = packedInfo.find(key);
        if (pit != packedInfo.end()) pit->second.words_per_entry = words_per_entry;
    }

    const uint64_t* getInstructionTable(uint64_t airgroupId, uint64_t airId) {
        auto it = instrTables.find({airgroupId, airId});
        return (it != instrTables.end() && !it->second.empty()) ? it->second.data() : nullptr;
    }

    uint64_t getInstructionTableEntries(uint64_t airgroupId, uint64_t airId) {
        auto it = instrEntries.find({airgroupId, airId});
        return it != instrEntries.end() ? it->second : 0;
    }

    PackedInfoCPU* getPackedInfo(uint64_t airgroupId, uint64_t airId) {
        if (!packedTrace) return nullptr;

        auto it = packedInfo.find({airgroupId, airId});
        if (it != packedInfo.end())
            return &it->second;
        return nullptr;
    }

    // Indexed cm1 unpack (row-major dst, matching unpack_cpu): compact rows + shared
    // instruction table reconstruct the full nCols output per row.
    void unpack_cpu_indexed(
        const uint64_t* src,
        const uint64_t* table,
        uint64_t* dst,
        uint64_t nRows,
        uint64_t nCols,
        uint64_t words_per_row,
        uint64_t words_per_entry,
        const std::vector<uint64_t> &unpack_info,
        const std::vector<uint8_t> &col_source,
        uint64_t index_bits,
        uint64_t num_entries,
        uint64_t airgroupId,
        uint64_t airId
    ) {
        for (uint64_t row = 0; row < nRows; row++) {
            const uint64_t* rbase = &src[row * words_per_row];
            uint64_t rword = rbase[0], ridx = 0, roff = 0;
            uint64_t index = cpu_idx_read_bits(rbase, words_per_row, rword, ridx, roff, index_bits);
            if (index >= num_entries) {
                zklog.error("unpack_cpu_indexed: air (" + std::to_string(airgroupId) + "," +
                            std::to_string(airId) + ") row " + std::to_string(row) +
                            " has instruction index " + std::to_string(index) +
                            " but the table only has " + std::to_string(num_entries) + " entries");
                exitProcess();
            }

            const uint64_t* tbase = &table[index * words_per_entry];
            uint64_t tword = tbase[0], tidx = 0, toff = 0;

            uint64_t* unpacked_row = &dst[row * nCols];
            for (uint64_t c = 0; c < nCols; c++) {
                uint64_t nbits = unpack_info[c];
                unpacked_row[c] = col_source[c]
                    ? cpu_idx_read_bits(tbase, words_per_entry, tword, tidx, toff, nbits)
                    : cpu_idx_read_bits(rbase, words_per_row, rword, ridx, roff, nbits);
            }
        }
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