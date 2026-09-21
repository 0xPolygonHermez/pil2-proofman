// COPY of pil2-stark/src/starkpil/recursion_trace/exec_layout.hpp, which owns this format.
// The witness library is built in a temp dir holding only setup/circom + the goldilocks
// sources, so it cannot include the original. Keep the two in step.
#ifndef EXEC_LAYOUT_HPP
#define EXEC_LAYOUT_HPP

// The `.exec` file's layout, shared by the map reader (exec_file.hpp) and the gate-band reader
// (gate_bands.hpp). Written by `write_exec_file` in plonk2pil/mod.rs, which owns the format.
//
//   [0] EXEC_MAGIC | EXEC_FORMAT_VERSION
//   [1] nAdds   [2] mapRows   [3] mapCols
//   additions: (sl, sr, coefL, coefR) each
//   map: mapRows * mapCols u32 entries, row-major, two per word, padded to a whole word
//   gate bands: version, count, (row, kind) per band
//
// The map covers only the rows and columns that carry placements -- the packers fill rows from 0
// and leave the power-of-two padding alone, and a gate band's interior columns are never mapped.
// Every cell outside the extent is zero.

#include <cstdint>

namespace exec_layout {

// "PXEC" in the high half. The pre-magic layout opened with nAdds, a small count, so no older
// file can be mistaken for one carrying this header.
constexpr uint64_t EXEC_MAGIC = 0x5058454300000000ull;
constexpr uint64_t EXEC_MAGIC_MASK = 0xFFFFFFFF00000000ull;

// Mirrors EXEC_FORMAT_VERSION in plonk2pil/mod.rs, which writes it.
constexpr uint64_t EXEC_FORMAT_VERSION = 2;

constexpr uint64_t HEADER_WORDS = 4;

struct Header {
    uint64_t nAdds = 0;
    uint64_t mapRows = 0;
    uint64_t mapCols = 0;
    uint64_t version = 0;   // version seen, for the error message
    bool magic = false;     // the buffer carries this header at all
    bool versionOk = false; // and its version is the one this build reads
    bool valid = false;     // and its dimensions describe a real layout
};

// Dimensions are checked here, not at each use, so `map_at` and `bands_at` are total functions on
// a valid header. An invalid one reports every dimension as 0, which reads as an empty map.
inline Header header(const uint64_t *exec, uint64_t execWords) {
    Header h;
    if (execWords < HEADER_WORDS) return h;
    if ((exec[0] & EXEC_MAGIC_MASK) != EXEC_MAGIC) return h;
    h.magic = true;
    h.version = exec[0] & ~EXEC_MAGIC_MASK;
    if (h.version != EXEC_FORMAT_VERSION) return h;
    h.versionOk = true;

    const uint64_t nAdds = exec[1], mapRows = exec[2], mapCols = exec[3];
    // A wrapped offset would point back inside the header, so refuse rather than compute it.
    if (nAdds > (UINT64_MAX - HEADER_WORDS) / 4) return h;
    if (mapCols != 0 && mapRows > UINT64_MAX / mapCols) return h;
    const uint64_t entries = mapRows * mapCols;
    if (entries / 2 + (entries & 1) > UINT64_MAX - (HEADER_WORDS + nAdds * 4)) return h;

    h.nAdds = nAdds;
    h.mapRows = mapRows;
    h.mapCols = mapCols;
    h.valid = true;
    return h;
}

// First word of the map.
inline uint64_t map_at(const Header &h) { return HEADER_WORDS + h.nAdds * 4; }

// First word past the map, where the gate-band section starts.
inline uint64_t bands_at(const Header &h) {
    const uint64_t entries = h.mapRows * h.mapCols;
    return map_at(h) + entries / 2 + (entries & 1);
}

}  // namespace exec_layout

#endif
