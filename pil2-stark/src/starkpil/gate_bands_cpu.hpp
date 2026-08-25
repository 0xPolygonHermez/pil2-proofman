#ifndef GATE_BANDS_CPU_HPP
#define GATE_BANDS_CPU_HPP

// CPU expander: fills every band's interior cells from its boundary cells.
//
// Runs after getCommitedPols has placed the boundary -- a band's input at its first row, key
// alongside, output at its last. Reads those back, runs the gate's permutation, and scatters
// the snapshots into the interior.

#include <cstdint>
#include "gate_bands.hpp"
#include "goldilocks_base_field.hpp"

#include "gate_bands_poseidon1.hpp"
#include "gate_bands_poseidon2.hpp"

namespace gate_bands {

// The snapshots a band needs, plus the permuted output, from its boundary cells.
inline void band_snapshots(const Goldilocks::Element *trace, uint64_t nCols, uint64_t row,
                           uint64_t kind, uint64_t *im, uint64_t *out) {
    uint64_t in[POS_W];
    for (int i = 0; i < POS_W; i++) in[i] = Goldilocks::toU64(trace[row * nCols + i]);
    const bool p1 = is_poseidon1(kind);
    if (!is_compression(kind)) {
        if (p1) poseidon1::snapshots(im, out, in, 0, false);
        else    poseidon2::snapshots(im, out, in, 0, false);
        return;
    }
    // Key bits sit where the setup puts them: a[16] and a[17] of a compressor band's first
    // row, a[15] of the two middle rows for aggregation.
    uint64_t k;
    if (is_aggregation(kind)) {
        k = (Goldilocks::toU64(trace[(row + 1) * nCols + 15]) & 1)
          | ((Goldilocks::toU64(trace[(row + 2) * nCols + 15]) & 1) << 1);
    } else {
        k = (Goldilocks::toU64(trace[row * nCols + 16]) & 1)
          | ((Goldilocks::toU64(trace[row * nCols + 17]) & 1) << 1);
    }
    if (p1) poseidon1::snapshots(im, out, in, k, true);
    else    poseidon2::snapshots(im, out, in, k, true);
}

// Compressor band: one chain slot, snapshots down rows 0..9, anchors on row 5.
inline void expand_compressor_band(Goldilocks::Element *trace, uint64_t nCols, uint64_t row,
                                   const uint64_t *im) {
    auto at = [&](uint64_t r, int c) -> Goldilocks::Element & { return trace[r * nCols + c]; };
    for (int k = 0; k < POS_ROWS; k++) {
        const int g = pos_row_im(k);
        if (g < 0) continue;
        for (int i = 0; i < POS_W; i++) at(row + k, POS_COL_P + i) = Goldilocks::fromU64(im[g * POS_W + i]);
    }
    const uint64_t r5 = row + POS_ANCHOR_ROW;
    for (int i = 0; i < 11; i++) at(r5, POS_COL_P + i) = Goldilocks::fromU64(im[5 * POS_W + i]);
    for (int i = 0; i < 5; i++) at(r5, POS_COL_P + 11 + i) = Goldilocks::fromU64(im[7 * POS_W + i]);
    for (int i = 0; i < 6; i++)
        at(r5, POS_ANCHOR_OVERFLOW_COL + i) = Goldilocks::fromU64(im[7 * POS_W + 5 + i]);
}

// Aggregation band: two chains, so a row carries two snapshots. anchors[0..15] run across
// chain 2 of the anchor row; anchors[16..21] spill to a[9..14] of that same row.
inline void expand_aggregation_band(Goldilocks::Element *trace, uint64_t nCols, uint64_t row,
                                    const uint64_t *im) {
    auto at = [&](uint64_t r, int c) -> Goldilocks::Element & { return trace[r * nCols + c]; };
    // (row offset, chain-1 group, chain-2 group); -1 means that slot is not a snapshot.
    constexpr int LAYOUT[4][3] = {{0, 0, 1}, {1, 2, 3}, {3, 8, 9}, {4, 10, 11}};
    for (auto &l : LAYOUT) {
        for (int i = 0; i < POS_W; i++) {
            at(row + l[0], AGG_COL_P1 + i) = Goldilocks::fromU64(im[l[1] * POS_W + i]);
            at(row + l[0], AGG_COL_P2 + i) = Goldilocks::fromU64(im[l[2] * POS_W + i]);
        }
    }
    const uint64_t ra = row + AGG_ANCHOR_ROW;
    for (int i = 0; i < POS_W; i++) at(ra, AGG_COL_P1 + i) = Goldilocks::fromU64(im[4 * POS_W + i]);
    // anchors: 11 from im[5], then 11 from im[7]; the first 16 across chain 2, the last 6 at
    // a[9..14] -- never a[15], which carries a key bit on this row.
    auto anchor = [&](int idx) { return idx <= 10 ? im[5 * POS_W + idx] : im[7 * POS_W + (idx - 11)]; };
    for (int idx = 0; idx < 16; idx++) at(ra, AGG_COL_P2 + idx) = Goldilocks::fromU64(anchor(idx));
    for (int idx = 16; idx < 22; idx++)
        at(ra, AGG_ANCHOR_OVERFLOW_COL + (idx - 16)) = Goldilocks::fromU64(anchor(idx));
}

// Compares the permuted state against the output the map already placed at the band's last
// row, columns 0..15. Returns the first differing limb, or POS_W when they agree.
inline uint64_t output_mismatch(const Goldilocks::Element *trace, uint64_t nCols, uint64_t row,
                                uint64_t kind, const uint64_t *out) {
    const uint64_t outRow = row + (is_aggregation(kind) ? AGG_OUT_ROW : (uint64_t)(POS_ROWS - 1));
    for (int i = 0; i < POS_W; i++) {
        if (Goldilocks::toU64(trace[outRow * nCols + i]) != out[i]) return (uint64_t)i;
    }
    return (uint64_t)POS_W;
}

inline bool expand_poseidon_band(Goldilocks::Element *trace, uint64_t nCols, uint64_t row, uint64_t kind) {
    uint64_t im[POS_IM_GROUPS * POS_W] = {0}, out[POS_W] = {0};
    band_snapshots(trace, nCols, row, kind, im, out);
    // Disagreement means the boundary and this permutation describe different hashes: either
    // the gate and this copy have drifted, or the band record points at the wrong row.
    if (output_mismatch(trace, nCols, row, kind, out) != (uint64_t)POS_W) return false;
    if (is_aggregation(kind)) {
        expand_aggregation_band(trace, nCols, row, im);
    } else {
        expand_compressor_band(trace, nCols, row, im);
    }
    return true;
}

// Why an expansion stopped. Reported rather than logged so this header needs nothing beyond
// the field; the API boundary owns the logging and the abort.
enum class ExpandStatus {
    Ok = 0,
    MalformedSection,       // the band section does not describe this buffer
    UnsupportedVersion,     // the key's section layout is not the one this build reads
    UnsupportedExecFormat,  // the enclosing exec file's layout is not the one this build reads
    UnexpandableBand,       // unknown kind, or a band that would run off the end of the trace
    OutputMismatch,         // the band's own input does not hash to the output already in place
};

struct ExpandResult {
    uint64_t nBands = 0;    // bands in the section (all expanded when status is Ok)
    uint64_t badBand = 0;   // index of the offending band, meaningful only for the band errors
    uint64_t row = 0;
    uint64_t kind = 0;
    uint64_t version = 0;   // section version seen, for the version error
    ExpandStatus status = ExpandStatus::Ok;
};

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

    const uint64_t bad = first_bad_band(b, nBands, nRows);
    if (bad != nBands) {
        res.status = ExpandStatus::UnexpandableBand;
        res.badBand = bad;
        res.row = b[bad * 2];
        res.kind = b[bad * 2 + 1];
        return res;
    }

    // Bands never overlap -- the setup lays them out end to end -- so they fill independently.
    uint64_t firstMismatch = nBands;
#pragma omp parallel for
    for (uint64_t i = 0; i < nBands; i++) {
        if (!expand_poseidon_band(trace, nCols, b[i * 2], b[i * 2 + 1])) {
#pragma omp critical
            if (i < firstMismatch) firstMismatch = i;
        }
    }
    if (firstMismatch != nBands) {
        res.status = ExpandStatus::OutputMismatch;
        res.badBand = firstMismatch;
        res.row = b[firstMismatch * 2];
        res.kind = b[firstMismatch * 2 + 1];
    }
    return res;
}

}  // namespace gate_bands

#endif
