#ifndef LIB_API_INTERNAL_H
#define LIB_API_INTERNAL_H
#include "starks_api.hpp"
#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include <utility>


#include "unpack_indexed_row.hpp"
#include "hash_family.hpp"
#include "poseidon_goldilocks.hpp"
#include "poseidon2_goldilocks.hpp"
#include "blake3_goldilocks.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "pack_columns.hpp"

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
    std::vector<uint8_t> col_lane;   // per column: lane whose index selects its entry
    uint64_t index_bits = 0;
    uint64_t words_per_entry = 0;
    uint64_t lanes = 0;              // indices per row; 0/1 is the single-lane shape
    bool indexed() const { return !col_source.empty(); }
};

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
                          uint8_t* col_lane_, uint64_t index_bits, uint64_t words_per_entry,
                          uint64_t lanes) {
        if (!is_packed) return;
        std::vector<uint64_t> unpack_vec(unpack_info_, unpack_info_ + nCols);
        std::vector<uint8_t> col_source_vec, col_lane_vec;
        if (col_source_ != nullptr) {
            col_source_vec.assign(col_source_, col_source_ + nCols);
            // Lane-less is the single-lane shape. Above one lane the map is load-bearing:
            // without it every column would decode from lane 0's entry, a wrong trace with
            // no other symptom, so refuse it the way the GPU slot path does.
            if (col_lane_ != nullptr) {
                col_lane_vec.assign(col_lane_, col_lane_ + nCols);
            } else if (lanes > 1) {
                zklog.error("addPackedInfoCPU: air (" + std::to_string(airgroupId) + "," +
                            std::to_string(airId) + ") packs " + std::to_string(lanes) +
                            " lanes per row but carries no col_lane map");
                exitProcess();
            } else {
                col_lane_vec.assign(nCols, 0);
            }
        }
        PackedInfoCPU pInfo = {is_packed,     num_packed_words, unpack_vec, col_source_vec,
                               col_lane_vec,  index_bits,       words_per_entry, lanes};
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

    // Indexed cm1 unpack (row-major dst, matching unpack_cpu). The per-row walk lives in
    // unpack_indexed_row.hpp, shared with its unit test; this adds the fatal report an
    // out-of-range index deserves on the host (a kernel cannot abort).
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
        const std::vector<uint8_t> &col_lane,
        uint64_t index_bits,
        uint64_t lanes,
        uint64_t num_entries,
        uint64_t airgroupId,
        uint64_t airId
    ) {
        for (uint64_t row = 0; row < nRows; row++) {
            uint64_t bad_lane = 0, bad_index = 0;
            if (!unpackIndexedRow(&src[row * words_per_row], words_per_row, table, words_per_entry,
                                  num_entries, index_bits, lanes, unpack_info.data(),
                                  col_source.data(), col_lane.data(), nCols, &dst[row * nCols],
                                  &bad_lane, &bad_index)) {
                zklog.error("unpack_cpu_indexed: air (" + std::to_string(airgroupId) + "," +
                            std::to_string(airId) + ") row " + std::to_string(row) + " lane " +
                            std::to_string(bad_lane) + " has instruction index " +
                            std::to_string(bad_index) + " but the table only has " +
                            std::to_string(num_entries) + " entries");
                exitProcess();
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
        unpackRowsBits(src, dst, nRows, nCols, unpack_info.data(), words_per_row);
    }
};

#endif