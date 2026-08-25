#ifndef GATE_BANDS_HPP
#define GATE_BANDS_HPP

// Rebuilding a hash gate's interior trace cells from its boundary cells.
//
// `getCommitedPols` can only place values the circom witness carries. A gate's interiors are a
// pure function of its inputs, and those inputs are boundary cells the map already places, so
// they are recomputed here rather than routed through the witness.
//
// A band's layout is fixed per gate shape, so the exec file records only where each band starts
// and which shape it is. See GateBand in plonk2pil/r1cs/types.rs.

#include <cstdint>
#include <climits>
#include "goldilocks_base_field.hpp"
#include "exec_layout.hpp"

// Mirrors GateBandKind. Serialized, so these are a wire format: append, never renumber.
enum GateBandKind : uint64_t {
    GB_POSEIDON1_COMPRESSOR_SPONGE = 1,
    GB_POSEIDON1_COMPRESSOR_COMPRESSION = 2,
    GB_POSEIDON1_AGGREGATION_SPONGE = 3,
    GB_POSEIDON1_AGGREGATION_COMPRESSION = 4,
    GB_POSEIDON2_COMPRESSOR_SPONGE = 5,
    GB_POSEIDON2_COMPRESSOR_COMPRESSION = 6,
    GB_POSEIDON2_AGGREGATION_SPONGE = 7,
    GB_POSEIDON2_AGGREGATION_COMPRESSION = 8,
};

// Band geometry: a property of the setup type, not the family. A compressor band is 10 rows
// with one chain slot, an aggregation band 5 rows with two; the family only picks which
// permutation fills them. im[k] is the k-th snapshot, laid out in gate_bands_poseidon1.hpp.
namespace gate_bands {

constexpr int POS_W = 16;
constexpr int POS_IM_GROUPS = 12;

// Compressor: chain at a[30..45], anchors on row 5 with the tail spilling to a[18..23].
constexpr int POS_ROWS = 10;
constexpr int POS_COL_P = 30;
constexpr int POS_ANCHOR_ROW = 5;
constexpr int POS_ANCHOR_OVERFLOW_COL = 18;

// Aggregation: two chains at a[16..31] and a[32..47], output at row 4, anchors on row 2
// across chain 2 with the tail at a[9..14]. a[15] of the two middle rows holds the key, so
// the tail deliberately stops short of it.
constexpr int AGG_ROWS = 5;
constexpr int AGG_COL_P1 = 16;
constexpr int AGG_COL_P2 = 32;
constexpr int AGG_OUT_ROW = 4;
constexpr int AGG_ANCHOR_ROW = 2;
constexpr int AGG_ANCHOR_OVERFLOW_COL = 9;

// Compressor geometry, row offset -> snapshot group; -1 for the anchor row, which holds
// partial-round anchors instead. BandSink::groupSlot walks the same mapping in reverse.
// A function rather than an array so device code can use it without a __device__ copy.
constexpr int pos_row_im(int row_off) {
    switch (row_off) {
        case 0: return 0;
        case 1: return 1;
        case 2: return 2;
        case 3: return 3;
        case 4: return 4;
        case 5: return -1;   // anchor row
        case 6: return 8;
        case 7: return 9;
        case 8: return 10;
        case 9: return 11;
        default: return -1;
    }
}

// constexpr so device code can use these without a __device__ duplicate.
constexpr bool is_compression(uint64_t kind) {
    return kind == GB_POSEIDON1_COMPRESSOR_COMPRESSION || kind == GB_POSEIDON1_AGGREGATION_COMPRESSION
        || kind == GB_POSEIDON2_COMPRESSOR_COMPRESSION || kind == GB_POSEIDON2_AGGREGATION_COMPRESSION;
}

constexpr bool is_aggregation(uint64_t kind) {
    return kind == GB_POSEIDON1_AGGREGATION_SPONGE || kind == GB_POSEIDON1_AGGREGATION_COMPRESSION
        || kind == GB_POSEIDON2_AGGREGATION_SPONGE || kind == GB_POSEIDON2_AGGREGATION_COMPRESSION;
}

constexpr bool is_poseidon1(uint64_t kind) {
    return kind >= GB_POSEIDON1_COMPRESSOR_SPONGE && kind <= GB_POSEIDON1_AGGREGATION_COMPRESSION;
}

// Every kind this build can expand. The predicates above would read an unrecognised one as a
// Poseidon2 compressor band and fill it with the wrong permutation, so callers reject instead.
constexpr bool is_known_kind(uint64_t kind) {
    return kind >= GB_POSEIDON1_COMPRESSOR_SPONGE && kind <= GB_POSEIDON2_AGGREGATION_COMPRESSION;
}

// Rows a band of this kind occupies.
constexpr int band_rows(uint64_t kind) { return is_aggregation(kind) ? AGG_ROWS : POS_ROWS; }

// Layout version of the band section; `band_section` refuses one it does not know. Bump on any
// layout change. Mirrored by GATE_BAND_FORMAT_VERSION in plonk2pil/mod.rs, which writes it.
//
// This cannot reach a reader that predates the section entirely -- such a reader stops at the
// map. That case is safe anyway: a key with bands has no interior placements, so a prover that
// skips the expansion leaves them zero and the AIR rejects the proof.
constexpr uint64_t GATE_BAND_FORMAT_VERSION = 1;

enum class BandSection {
    Absent,                 // no section past the map: nothing to expand
    Ok,
    Malformed,              // header or count does not describe this buffer
    UnsupportedVersion,     // the key's section layout is not the one this build reads
    UnsupportedExecFormat,  // the enclosing exec file's layout is not the one this build reads
};

struct BandsView {
    const uint64_t *bands = nullptr;
    uint64_t n = 0;
    uint64_t version = 0;
    BandSection status = BandSection::Absent;
};

// The band section of an exec buffer: version, count, then (row, kind) per band. `execWords` is
// the buffer's true length, which is how absence is detected -- the prefix alone is a complete
// exec file. Where the section starts comes from the exec header, so this needs no trace width.
inline BandsView band_section(const uint64_t *exec, uint64_t execWords) {
    BandsView v;
    const exec_layout::Header h = exec_layout::header(exec, execWords);
    if (!h.valid) {
        // Without a header this build reads, the section's offset is unknown. A refused version
        // is reported apart from a corrupt header, since only one of them is worth regenerating.
        v.version = h.version;
        v.status = h.magic && !h.versionOk ? BandSection::UnsupportedExecFormat : BandSection::Malformed;
        return v;
    }
    const uint64_t prefix = exec_layout::bands_at(h);
    if (execWords <= prefix) return v;                        // Absent
    if (execWords < prefix + 2) {                             // version without a count
        v.status = BandSection::Malformed;
        return v;
    }
    v.version = exec[prefix];
    if (v.version != GATE_BAND_FORMAT_VERSION) {
        v.status = BandSection::UnsupportedVersion;
        return v;
    }
    const uint64_t count = exec[prefix + 1];
    // Two words per band must actually be there, or reading them runs off the buffer.
    if (count > (execWords - prefix - 2) / 2) {
        v.status = BandSection::Malformed;
        return v;
    }
    v.n = count;
    v.bands = &exec[prefix + 2];
    v.status = BandSection::Ok;
    return v;
}

// Index of the first band this expander cannot fill -- unknown kind, or one that would write
// past row `nRows` -- or `nBands` if they are all good.
inline uint64_t first_bad_band(const uint64_t *b, uint64_t nBands, uint64_t nRows) {
    for (uint64_t i = 0; i < nBands; i++) {
        const uint64_t kind = b[i * 2 + 1];
        if (!is_known_kind(kind)) return i;
        if (b[i * 2] + (uint64_t)band_rows(kind) > nRows) return i;
    }
    return nBands;
}

}  // namespace gate_bands

#endif
