#ifndef GATE_BANDS_CPU_HPP
#define GATE_BANDS_CPU_HPP

// CPU expander: fills every band's interior cells from its boundary cells.
//
// Runs after getCommitedPols has placed the boundary -- a band's input at its first row, key
// alongside, output at its last. This file parses the exec file's band section, validates it, and
// hands the whole list to the one back-end that owns it; the reconstruction lives per family in
// gate_bands_<family>_cpu.hpp. Adding a family means a back-end and an arm in the switch below.

#include <cstdint>
#include "gate_bands.hpp"
#include "gate_bands_blake3_cpu.hpp"
#include "gate_bands_poseidon_cpu.hpp"
#include "goldilocks_base_field.hpp"

namespace gate_bands {

// Every band in the exec buffer. A file without a band section is Ok with nBands == 0.
inline ExpandResult expand_gate_bands(Goldilocks::Element *trace, const uint64_t *exec, uint64_t nCols,
                                      uint64_t execWords, uint64_t nRows) {
    ExpandResult res;
    const BandsView view = band_section(exec, execWords);
    res.version = view.version;
    switch (view.status) {
        case BandSection::Absent: return res;
        case BandSection::Malformed:          res.status = ExpandStatus::MalformedSection;   return res;
        case BandSection::UnsupportedVersion: res.status = ExpandStatus::UnsupportedVersion; return res;
        case BandSection::UnsupportedExecFormat:
            res.status = ExpandStatus::UnsupportedExecFormat; return res;
        case BandSection::Ok: break;
    }

    const uint64_t *b = view.bands;
    const uint64_t nBands = view.n;
    res.nBands = nBands;

    // Kinds and row bounds checked once here, so a back-end may assume both.
    const uint64_t bad = first_bad_band(b, nBands, nRows);
    if (bad != nBands) {
        res.status = ExpandStatus::UnexpandableBand;
        res.badBand = bad;
        res.row = b[bad * 3];
        res.kind = b[bad * 3 + 1];
        return res;
    }

    switch (family_of_bands(b, nBands)) {
        case Family::None: break;
        case Family::Poseidon: poseidon_cpu::expand_all(trace, nCols, nRows, b, nBands, view.aux, res); break;
        case Family::Blake3:   blake3_cpu::expand_all(trace, nCols, nRows, b, nBands, view.aux, res);   break;
        case Family::Mixed:    res.status = ExpandStatus::MixedFamilies; break;
    }
    return res;
}

}  // namespace gate_bands

#endif
