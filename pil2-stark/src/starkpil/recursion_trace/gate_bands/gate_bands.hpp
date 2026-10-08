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
#include "../exec_layout.hpp"

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
    // 56-row blocks hosting LANES permutations. Neither LANES nor the band width is per band --
    // both are the air's, and both arrive packed in the band section's aux word: LANES in the low
    // 32 bits, the a[]/S[] band width in the high 32. The band is there because it is not a
    // constant (18 on the aggregator, 27 on the compressor) and it fixes where every lane column
    // starts.
    GB_BLAKE3_NODE = 9,
    GB_BLAKE3_COMPRESS_CHUNK = 10,
    GB_BLAKE3_COMPRESS_PARENT = 11,
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

constexpr bool is_blake3(uint64_t kind) {
    return kind >= GB_BLAKE3_NODE && kind <= GB_BLAKE3_COMPRESS_PARENT;
}

constexpr bool is_poseidon1(uint64_t kind) {
    return kind >= GB_POSEIDON1_COMPRESSOR_SPONGE && kind <= GB_POSEIDON1_AGGREGATION_COMPRESSION;
}

// Every kind this build can expand. The predicates above would read an unrecognised one as a
// Poseidon2 compressor band and fill it with the wrong permutation, so callers reject instead.
constexpr bool is_known_kind(uint64_t kind) {
    return kind >= GB_POSEIDON1_COMPRESSOR_SPONGE && kind <= GB_BLAKE3_COMPRESS_PARENT;
}

// Which back-end expands a band. A setup is ONE hash family, so this is an air-wide property --
// but it is derived from the bands rather than declared, so `Mixed` exists to be rejected: each
// back-end skips kinds it does not own, and a mixed list would leave the other family's bands
// unwritten rather than failing.
enum class Family : uint64_t {
    None = 0,      // no bands at all
    Poseidon = 1,  // P1 and P2 share a back-end: same geometry, different constants
    Blake3 = 2,
    Mixed = 3,     // never valid; a band list must be one family
};

// Only meaningful for a kind `is_known_kind` accepts.
constexpr Family family_of(uint64_t kind) { return is_blake3(kind) ? Family::Blake3 : Family::Poseidon; }

inline Family family_of_bands(const uint64_t *b, uint64_t nBands) {
    Family f = Family::None;
    for (uint64_t i = 0; i < nBands; i++) {
        const Family k = family_of(b[i * 3 + 1]);
        if (f == Family::None) f = k;
        else if (f != k) return Family::Mixed;
    }
    return f;
}

// Rows a band of this kind occupies.
constexpr int BLAKE3_ROWS = 56;
constexpr int band_rows(uint64_t kind) {
    if (is_blake3(kind)) return BLAKE3_ROWS;
    return is_aggregation(kind) ? AGG_ROWS : POS_ROWS;
}

// Layout version of the band section; `band_section` refuses one it does not know. Bump on any
// layout change. Mirrored by GATE_BAND_FORMAT_VERSION in plonk2pil/mod.rs, which writes it.
//
// This cannot reach a reader that predates the section entirely -- such a reader stops at the
// map. That case is safe anyway: a key with bands has no interior placements, so a prover that
// skips the expansion leaves them zero and the AIR rejects the proof.
constexpr uint64_t GATE_BAND_FORMAT_VERSION = 2;

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
    /// A per-air parameter the expander cannot infer: BLAKE3's LANES. Poseidon writes 0.
    uint64_t aux = 0;
    BandSection status = BandSection::Absent;
};

// The band section of an exec buffer: version, count, a per-air aux word, then
// (row, kind, payload) per band.
// The payload carries a per-block constant the expander cannot read off the witness trace --
// BLAKE3's `flags`, which the AIR holds in a fixed column. Poseidon kinds write 0. `execWords` is
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
    if (execWords < prefix + 3) {                             // version without a count and aux
        v.status = BandSection::Malformed;
        return v;
    }
    v.version = exec[prefix];
    if (v.version != GATE_BAND_FORMAT_VERSION) {
        v.status = BandSection::UnsupportedVersion;
        return v;
    }
    const uint64_t count = exec[prefix + 1];
    // Three words per band must actually be there, or reading them runs off the buffer.
    if (count > (execWords - prefix - 3) / 3) {
        v.status = BandSection::Malformed;
        return v;
    }
    v.n = count;
    v.aux = exec[prefix + 2];
    v.bands = &exec[prefix + 3];
    v.status = BandSection::Ok;
    return v;
}

// Why an expansion stopped. Reported rather than logged so the back-ends need nothing beyond the
// field; the API boundary owns the logging and the abort.
enum class ExpandStatus {
    Ok = 0,
    MalformedSection,       // the band section does not describe this buffer
    UnsupportedVersion,     // the key's section layout is not the one this build reads
    UnsupportedExecFormat,  // the enclosing exec file's layout is not the one this build reads
    UnexpandableBand,       // unknown kind, or a band that would run off the end of the trace
    OutputMismatch,         // the band's own input does not hash to the output already in place
    TableTooLargeForTrace,  // a BLAKE3 air whose trace cannot hold the 2^17-row lookup table
    MixedFamilies,          // one band list naming two hash families; no back-end owns all of it
};

struct ExpandResult {
    uint64_t nBands = 0;    // bands in the section (all expanded when status is Ok)
    uint64_t badBand = 0;   // index of the offending band, meaningful only for the band errors
    uint64_t row = 0;
    uint64_t kind = 0;
    uint64_t version = 0;   // section version seen, for the version error
    ExpandStatus status = ExpandStatus::Ok;
};

// Index of the first band this expander cannot fill -- unknown kind, or one that would write
// past row `nRows` -- or `nBands` if they are all good.
inline uint64_t first_bad_band(const uint64_t *b, uint64_t nBands, uint64_t nRows) {
    for (uint64_t i = 0; i < nBands; i++) {
        const uint64_t kind = b[i * 3 + 1];
        if (!is_known_kind(kind)) return i;
        if (b[i * 3] + (uint64_t)band_rows(kind) > nRows) return i;
    }
    return nBands;
}

}  // namespace gate_bands

#endif
