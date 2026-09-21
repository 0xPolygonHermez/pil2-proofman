#ifndef EXEC_FILE
#define EXEC_FILE

#include <cstring>
#include "goldilocks_base_field.hpp"
#include "exec_layout.hpp"

// Gathers the circom witness into the committed-polynomial trace. Cells outside the map's live
// extent (see exec_layout.hpp), and mapped cells holding the unused-signal sentinel 0, are zero.
void getCommitedPols(Goldilocks::Element *circomWitness, uint64_t *exec_data, Goldilocks::Element *witness, Goldilocks::Element* publics, uint64_t sizeWitness, uint64_t N, uint64_t nPublics, uint64_t nCommitedPols)  {

    // load_exec_file rejected any header this build cannot read, so the buffer is at least
    // HEADER_WORDS long. One that slipped through still degrades safely: the extent reads as
    // empty and the trace comes out zero rather than misparsed.
    const exec_layout::Header h = exec_layout::header(exec_data, exec_layout::HEADER_WORDS);
    const uint64_t *p_adds = &exec_data[exec_layout::HEADER_WORDS];
    // Entries are u32 pairs inside the u64 buffer, read through a byte pointer because aliasing
    // a uint64_t array as uint32_t is undefined. Compiles to the same load.
    const char *p_sMap = reinterpret_cast<const char *>(&exec_data[exec_layout::map_at(h)]);

    // The loader rejects a map wider than the trace but cannot check the height, not knowing
    // nBits. Clamp both so a bad key cannot walk off either buffer.
    const uint64_t mapRows = h.mapRows < N ? h.mapRows : N;
    const uint64_t mapCols = h.mapCols < nCommitedPols ? h.mapCols : nCommitedPols;

    for(uint64_t i = 0; i < nPublics; ++i) {
        publics[i] = circomWitness[1 + i];
    }

    // Serial: an addition may read a signal an earlier one produced.
    for (uint64_t i = 0; i < h.nAdds; i++) {
        uint64_t idx_1 = p_adds[i * 4];
        uint64_t idx_2 = p_adds[i * 4 + 1];

        Goldilocks::Element c = circomWitness[idx_1] * Goldilocks::fromU64(p_adds[i * 4 + 2]);
        Goldilocks::Element d = circomWitness[idx_2] * Goldilocks::fromU64(p_adds[i * 4 + 3]);
        circomWitness[sizeWitness + i] = c + d;
    }

#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < N; i++) {
        Goldilocks::Element *row = &witness[i * nCommitedPols];
        const uint64_t mapped = i < mapRows ? mapCols : 0;
        for (uint64_t j = 0; j < mapped; j++) {
            uint32_t idx;
            memcpy(&idx, p_sMap + (i * h.mapCols + j) * sizeof(uint32_t), sizeof(uint32_t));
            row[j] = idx != 0 ? circomWitness[idx] : Goldilocks::zero();
        }
        // Element is a bare uint64_t whose zero is all-zero bits.
        memset(row + mapped, 0, (nCommitedPols - mapped) * sizeof(Goldilocks::Element));
    }
}

#endif
