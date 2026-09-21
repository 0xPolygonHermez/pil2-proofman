#ifndef GATE_BANDS_BLAKE3_CPU_HPP
#define GATE_BANDS_BLAKE3_CPU_HPP

// Host driver for the BLAKE3 band kinds: the per-air preconditions and the multiplicity lifecycle.
// The expansion itself is in gate_bands_blake3.hpp, shared verbatim with the device back-end.

#include <cstdint>
#include "gate_bands.hpp"
#include "gate_bands_blake3.hpp"
#include "goldilocks_base_field.hpp"

namespace gate_bands {
namespace blake3_cpu {

// Every BLAKE3 band in a validated list, plus the lookup counts they make.
//
// Bands never overlap -- the setup lays them out end to end -- so they fill independently. The
// multiplicities do NOT: every lane's lookups land in one shared table. Each thread accumulates
// into its own copy and reduces once at the end, rather than an atomic per lookup -- at 16 XOR
// lookups per lane per row over 56 rows a block, atomics would dominate.
inline void expand_all(Goldilocks::Element *trace, uint64_t nCols, uint64_t nRows,
                       const uint64_t *b, uint64_t nBands, uint64_t aux, ExpandResult &res) {
    // LANES low, band width high. Both are required: a zero band would silently fall back to some
    // constant and put every lane column at the wrong offset, which is a corrupt trace rather than
    // a failure, so it is rejected here like a missing LANES.
    const uint64_t lanes = aux & 0xFFFFFFFFull;
    const uint64_t band = aux >> 32;
    if (lanes == 0 || band == 0) {
        res.status = ExpandStatus::MalformedSection;  // LANES and the band must travel with the air
        return;
    }
    // Counters past the trace would be dropped by write_multiplicities; refuse, as the GPU does.
    if (nRows < blake3::TABLE_SIZE) {
        res.status = ExpandStatus::TableTooLargeForTrace;
        return;
    }

    blake3::Multiplicities total;
#pragma omp parallel
    {
        // 1.5 MB per thread, so threads that draw no bands skip the reduction rather than pay it.
        blake3::Multiplicities local;
        bool touched = false;
#pragma omp for nowait
        for (uint64_t i = 0; i < nBands; i++) {
            const uint64_t row = b[i * 3], kind = b[i * 3 + 1], payload = b[i * 3 + 2];
            const blake3::Kind k = kind == GB_BLAKE3_NODE           ? blake3::Kind::Node
                                 : kind == GB_BLAKE3_COMPRESS_CHUNK ? blake3::Kind::Chunk
                                                                    : blake3::Kind::Parent;
            blake3::HostSink sink{local};
            blake3::expand_block(trace, nCols, row, lanes, band, k, payload, sink);
            touched = true;
        }
        if (touched) {
#pragma omp critical
            {
                for (uint64_t j = 0; j < blake3::TABLE_SIZE; j++) total.table[j] += local.table[j];
                for (uint64_t j = 0; j < blake3::RANGE_SIZE; j++) total.range[j] += local.range[j];
            }
        }
    }
    blake3::write_multiplicities(trace, nCols, nRows, lanes, band, total);
}

}  // namespace blake3_cpu
}  // namespace gate_bands

#endif
