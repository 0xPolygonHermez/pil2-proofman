//! plonk2pil: Convert R1CS constraint systems to PIL (Polynomial Identity Language).
//!
//! This module is the Rust port of the `recurser-js` pipeline.
//! It reads R1CS binary files, converts constraints to PLONK format, and runs
//! one of several setup routines to produce PIL source and fixed polynomials.
//!
//! The main entry point is [`plonk2pil`], which dispatches to the appropriate
//! setup variant based on the `setup_type` argument.

pub mod estimate;
pub mod merge_copies;
pub mod packers;
pub mod r1cs;
pub mod setups;
pub mod utils;

// Re-export old flat module names as aliases for backward compatibility
pub use r1cs::reader as r1cs_reader;
pub use r1cs::to_plonk as r1cs2plonk;
pub use r1cs::types as r1cs_types;
pub use setups::poseidon1::aggregation as aggregation_setup;
pub use setups::poseidon1::compressor as compressor_setup;

use anyhow::{bail, Result};

use r1cs::types::{read_r1cs_from_bytes, PlonkOptions};
pub use r1cs::types::{FixedPol, SetupResult};

/// The result returned by [`plonk2pil`], containing everything needed
/// for downstream proof generation.
#[derive(Debug, Clone)]
pub struct PlonkResult {
    /// Execution buffer: serialized additions and signal map.
    pub exec: Vec<u64>,
    /// Generated PIL source string.
    pub pil_str: String,
    /// Fixed polynomial values, as a flat list of (name, index, values).
    pub fixed_pols: Vec<FixedPol>,
    /// log2(number of rows).
    pub n_bits: usize,
    /// log2(rows) the circuit would take on its own, before any `min_n_bits` floor. Whether a
    /// circuit is intrinsically too big for recursive1 is a question about the circuit, so the
    /// threshold is read off this rather than off the pinned size.
    pub n_bits_natural: usize,
    /// Number of rows actually used before power-of-2 padding — mirrors JS `NUsed`.
    /// Used by the caller to compute the minimum nQueries for small recursive circuits (A2).
    pub n_used: usize,
    /// Airgroup name used in the PIL.
    pub airgroup_name: String,
    /// Air name used in the PIL.
    pub air_name: String,
}

/// Magic in the exec file's first word, tagging the layout below. The pre-magic layout opened
/// with `n_adds`, a small count, so no older file can be mistaken for one carrying this header.
pub const EXEC_MAGIC: u64 = 0x5058_4543_0000_0000; // "PXEC" in the high half

/// Layout version of the exec file, in the low half of the first word. Bump on any change to
/// the header, the map or their order. Mirrored by `exec_layout::EXEC_FORMAT_VERSION` in
/// pil2-stark/src/starkpil/exec_layout.hpp, which reads it.
pub const EXEC_FORMAT_VERSION: u64 = 2;

/// Words ahead of the additions: `magic|version`, `n_adds`, `map_rows`, `map_cols`.
pub const EXEC_HEADER_WORDS: usize = 4;

/// Layout version of the gate-band section. Must match `GATE_BAND_FORMAT_VERSION` in
/// pil2-stark/src/starkpil/gate_bands.hpp, which reads it.
pub const GATE_BAND_FORMAT_VERSION: u64 = 2;

/// Serialize PLONK additions and the signal map into an exec buffer.
///
/// Layout, all u64 LE:
/// - `[0]`: [`EXEC_MAGIC`] | [`EXEC_FORMAT_VERSION`]
/// - `[1]`: number of additions
/// - `[2]`: mapped rows, `[3]`: mapped columns
/// - additions: `(sl, sr, coef_l, coef_r)` each
/// - map: `map_rows * map_cols` u32 entries, row-major, two to a word, padded to a whole word
/// - gate bands: format version, band count, a per-air aux word, then `(row, kind, payload)` per band
///
/// The map is stored at its live extent, not the trace's: the packers fill rows from 0 and leave
/// the power-of-two padding untouched, and the columns a gate band fills are never mapped -- the
/// expander writes those from the band's boundary. `getCommitedPols` zeroes everything outside the
/// extent, which is what those cells held anyway. Both bounds are measured rather than assumed, so
/// a packer that starts using a row or column cannot silently have it dropped.
fn write_exec_file(
    adds: &[r1cs::to_plonk::PlonkAddition],
    s_map: &[Vec<u32>],
    gate_bands: &[r1cs::types::GateBand],
    band_aux: u64,
) -> Vec<u64> {
    let n_adds = adds.len();
    let all_cols = s_map.len();
    let all_rows = if all_cols > 0 { s_map[0].len() } else { 0 };
    debug_assert!(s_map.iter().all(|c| c.len() == all_rows), "s_map columns must all be the trace height");

    // Live extent: the last column and row carrying any placement. `rposition` scans from the end,
    // so on the usual shape -- dead columns and padding rows, both suffixes -- it stops early.
    let map_cols = (0..all_cols).rposition(|c| s_map[c].iter().any(|&v| v != 0)).map_or(0, |c| c + 1);
    let map_rows = (0..all_rows).rposition(|r| (0..map_cols).any(|c| s_map[c][r] != 0)).map_or(0, |r| r + 1);

    // A band's input sits at its first row in the low columns, so trimming cannot reach it.
    // Asserted rather than trusted: if it ever did, the prover would gather zeros for that input
    // and the failure would surface layers away as an unverifiable proof.
    if let Some(b) = gate_bands.iter().find(|b| b.row as usize >= map_rows) {
        panic!(
            "gate band at row {} lies outside the map's live extent of {} rows; its boundary \
             cells would be gathered as zero",
            b.row, map_rows
        );
    }

    let map_at = EXEC_HEADER_WORDS + n_adds * 4;
    let bands_at = map_at + (map_rows * map_cols).div_ceil(2);
    let mut buff = vec![0u64; bands_at + 3 + gate_bands.len() * 3];

    buff[0] = EXEC_MAGIC | EXEC_FORMAT_VERSION;
    buff[1] = n_adds as u64;
    buff[2] = map_rows as u64;
    buff[3] = map_cols as u64;

    for (i, add) in adds.iter().enumerate() {
        let at = EXEC_HEADER_WORDS + i * 4;
        buff[at] = add[0];
        buff[at + 1] = add[1];
        buff[at + 2] = add[2];
        buff[at + 3] = add[3];
    }

    // Two u32 entries per word, low half first: the order the bytes come out in on a little-endian
    // host, which this pipeline already assumes end to end (`to_le_bytes` out, raw bytes back into
    // u64s in). So the reader reads entries directly instead of unpacking them.
    for (c, column) in s_map[..map_cols].iter().enumerate() {
        for (r, &signal) in column[..map_rows].iter().enumerate() {
            let entry = r * map_cols + c;
            buff[map_at + entry / 2] |= (signal as u64) << (32 * (entry % 2));
        }
    }

    buff[bands_at] = GATE_BAND_FORMAT_VERSION;
    buff[bands_at + 1] = gate_bands.len() as u64;
    buff[bands_at + 2] = band_aux;
    for (i, b) in gate_bands.iter().enumerate() {
        buff[bands_at + 3 + i * 3] = b.row as u64;
        buff[bands_at + 3 + i * 3 + 1] = b.kind as u64;
        buff[bands_at + 3 + i * 3 + 2] = b.payload;
    }

    buff
}

/// Read an R1CS file and run the specified setup to produce PIL and fixed polynomials.
///
/// # Arguments
/// * `r1cs_data` - Raw bytes of the R1CS binary file.
/// * `setup_type` - One of `"compressor"`, `"aggregation"`.
/// * `options` - Optional configuration (airgroup name, max constraint degree).
///
/// # Returns
/// A [`PlonkResult`] containing the exec buffer, PIL source, and fixed polynomials.
pub fn plonk2pil(r1cs_data: &[u8], setup_type: &str, options: &PlonkOptions) -> Result<PlonkResult> {
    if !["compressor", "aggregation"].contains(&setup_type) {
        bail!("Invalid setup type: '{}'. Must be one of: compressor, aggregation", setup_type);
    }

    let r1cs = read_r1cs_from_bytes(r1cs_data)?;

    let res: SetupResult = match setup_type {
        "compressor" => packers::pack_compressor(&r1cs, options),
        "aggregation" => packers::pack_aggregation(&r1cs, options),
        _ => unreachable!(),
    };

    let exec = write_exec_file(&res.plonk_additions, &res.s_map, &res.gate_bands, res.band_aux);

    Ok(PlonkResult {
        exec,
        pil_str: res.pil_str,
        fixed_pols: res.fixed_pols,
        n_bits: res.n_bits,
        n_bits_natural: res.n_bits_natural,
        n_used: res.n_used,
        airgroup_name: res.airgroup_name,
        air_name: res.air_name,
    })
}

#[cfg(test)]
mod tests {
    use super::r1cs::to_plonk::*;
    use super::r1cs::types::read_r1cs_from_bytes;
    use super::*;

    /// Run the real compressor packer end-to-end on an r1cs (exercises the row-count
    /// assert + verify_merge_soundness). ESTIMATE_HASH selects Poseidon1 (default) or
    /// Poseidon2.
    ///   ESTIMATE_R1CS=/path/x.r1cs [ESTIMATE_HASH=Poseidon2] \
    ///     cargo test -p stark-recurser run_compressor --release -- --ignored --nocapture
    #[test]
    #[ignore]
    fn run_compressor() {
        use proofman_common::hash_family::GateRole;
        let Ok(f) = std::env::var("ESTIMATE_R1CS") else {
            eprintln!("set ESTIMATE_R1CS=/path/to/file.r1cs");
            return;
        };
        let bytes = std::fs::read(&f).unwrap_or_else(|e| panic!("read {f}: {e}"));
        let hash_id = std::env::var("ESTIMATE_HASH").unwrap_or_else(|_| "Poseidon1".into());
        let opts = PlonkOptions {
            airgroup_name: Some("Compressor".into()),
            max_constraint_degree: Some(5),
            hash_id,
            merge_copies: true,
            blake3_lanes: None,
        min_n_bits: None,
        };
        let res = plonk2pil(&bytes, "compressor", &opts).expect("compressor packing failed");
        let r1cs = read_r1cs_from_bytes(&bytes).unwrap();
        let cgi = get_custom_gates_info(&r1cs);
        let n_pos = cgi.n(GateRole::PoseidonCompression) + cgi.n(GateRole::PoseidonSponge);
        eprintln!("\n=== {f}  compressor OK: nBits={} nUsed={} n_pos={}", res.n_bits, res.n_used, n_pos);
    }

    /// Build a minimal R1CS with a single multiplication constraint and no custom gates.
    fn build_simple_r1cs_bytes() -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        buf.extend_from_slice(b"r1cs");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // 2 sections

        // Header
        let mut hdr: Vec<u8> = Vec::new();
        hdr.extend_from_slice(&8u32.to_le_bytes());
        hdr.extend_from_slice(&0xFFFF_FFFF_0000_0001u64.to_le_bytes());
        hdr.extend_from_slice(&4u32.to_le_bytes()); // nVars
        hdr.extend_from_slice(&1u32.to_le_bytes()); // nOutputs
        hdr.extend_from_slice(&1u32.to_le_bytes()); // nPubInputs
        hdr.extend_from_slice(&1u32.to_le_bytes()); // nPrvInputs
        hdr.extend_from_slice(&4u64.to_le_bytes()); // nLabels
        hdr.extend_from_slice(&1u32.to_le_bytes()); // nConstraints

        buf.extend_from_slice(&1u32.to_le_bytes()); // Section type 1 = header
        buf.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        buf.extend_from_slice(&hdr);

        // Constraint: wire_1 * wire_2 = wire_3
        let mut cdata: Vec<u8> = Vec::new();
        // A: 1 term (wire=1, coeff=1)
        cdata.extend_from_slice(&1u32.to_le_bytes());
        cdata.extend_from_slice(&1u32.to_le_bytes());
        cdata.extend_from_slice(&1u64.to_le_bytes());
        // B: 1 term (wire=2, coeff=1)
        cdata.extend_from_slice(&1u32.to_le_bytes());
        cdata.extend_from_slice(&2u32.to_le_bytes());
        cdata.extend_from_slice(&1u64.to_le_bytes());
        // C: 1 term (wire=3, coeff=1)
        cdata.extend_from_slice(&1u32.to_le_bytes());
        cdata.extend_from_slice(&3u32.to_le_bytes());
        cdata.extend_from_slice(&1u64.to_le_bytes());

        buf.extend_from_slice(&2u32.to_le_bytes()); // Section type 2 = constraints
        buf.extend_from_slice(&(cdata.len() as u64).to_le_bytes());
        buf.extend_from_slice(&cdata);

        buf
    }

    #[test]
    fn test_r1cs2plonk_basic() {
        let data = build_simple_r1cs_bytes();
        let r1cs = read_r1cs_from_bytes(&data).unwrap();
        let (constraints, additions) = r1cs2plonk(&r1cs);

        assert_eq!(constraints.len(), 1);
        assert!(additions.is_empty());
        // Should be a multiplication gate: qM != 0
        assert_ne!(constraints[0][3], 0);
    }

    /// One map entry, by row and column, out of the packed u32 pairs.
    fn map_entry(exec: &[u64], map_at: usize, map_cols: usize, row: usize, col: usize) -> u32 {
        let entry = row * map_cols + col;
        (exec[map_at + entry / 2] >> (32 * (entry % 2))) as u32
    }

    #[test]
    fn test_write_exec_file_roundtrip() {
        let adds: Vec<PlonkAddition> = vec![[10, 20, 30, 40], [50, 60, 70, 80]];
        let s_map: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];

        let exec = write_exec_file(&adds, &s_map, &[], 0);

        assert_eq!(exec[0], EXEC_MAGIC | EXEC_FORMAT_VERSION, "magic and version lead");
        assert_eq!(exec[1], 2, "2 additions");
        assert_eq!(exec[2], 4, "4 mapped rows");
        assert_eq!(exec[3], 2, "2 mapped columns");

        // First addition
        assert_eq!(exec[4], 10);
        assert_eq!(exec[5], 20);
        assert_eq!(exec[6], 30);
        assert_eq!(exec[7], 40);

        // Second addition
        assert_eq!(exec[8], 50);
        assert_eq!(exec[9], 60);
        assert_eq!(exec[10], 70);
        assert_eq!(exec[11], 80);

        // The map is row-major over (row, col), transposing the column-major s_map.
        let map_at = EXEC_HEADER_WORDS + adds.len() * 4;
        for (col, column) in s_map.iter().enumerate() {
            for (row, &signal) in column.iter().enumerate() {
                assert_eq!(map_entry(&exec, map_at, 2, row, col), signal, "row {row} col {col}");
            }
        }
    }

    /// The map is stored at its live extent: the trailing rows the power-of-two padding leaves
    /// empty, and the trailing columns a gate band fills instead of the map, are not written at
    /// all. Both are measured from the data, so a packer that starts using them keeps working.
    #[test]
    fn write_exec_file_trims_the_dead_rows_and_columns() {
        // 4 columns x 5 rows, but only columns 0..2 and rows 0..3 carry anything.
        let s_map: Vec<Vec<u32>> =
            vec![vec![1, 2, 3, 0, 0], vec![4, 0, 5, 0, 0], vec![0, 0, 0, 0, 0], vec![0, 0, 0, 0, 0]];

        let exec = write_exec_file(&[], &s_map, &[], 0);

        assert_eq!(exec[2], 3, "rows 3 and 4 hold nothing");
        assert_eq!(exec[3], 2, "columns 2 and 3 hold nothing");

        let map_at = EXEC_HEADER_WORDS;
        let expected = [[1u32, 4], [2, 0], [3, 5]];
        for (row, cols) in expected.iter().enumerate() {
            for (col, want) in cols.iter().enumerate() {
                assert_eq!(map_entry(&exec, map_at, 2, row, col), *want, "row {row} col {col}");
            }
        }

        // 6 entries = 3 words, so the band section starts right after them.
        assert_eq!(exec[map_at + 3], GATE_BAND_FORMAT_VERSION);
        assert_eq!(exec[map_at + 4], 0, "no bands");
        assert_eq!(exec[map_at + 5], 0, "the per-air aux word, present even with no bands");
        assert_eq!(exec.len(), map_at + 6);
    }

    /// An odd entry count leaves half a word unused. The band section still has to start on a
    /// word boundary, or every reader after it is off by half an entry.
    #[test]
    fn write_exec_file_pads_an_odd_map_to_a_whole_word() {
        let s_map: Vec<Vec<u32>> = vec![vec![7]]; // 1 row x 1 col = one entry
        let exec = write_exec_file(&[], &s_map, &[], 0);

        assert_eq!((exec[2], exec[3]), (1, 1));
        assert_eq!(exec[EXEC_HEADER_WORDS], 7, "the entry, with the high half unused");
        assert_eq!(exec[EXEC_HEADER_WORDS + 1], GATE_BAND_FORMAT_VERSION, "section starts a word later");
        assert_eq!(exec.len(), EXEC_HEADER_WORDS + 4, "version, count and aux");
    }

    /// An all-zero map has no live extent at all and must not produce a negative or wrapped size.
    #[test]
    fn write_exec_file_handles_an_empty_map() {
        for s_map in [vec![], vec![vec![0u32; 4]; 3]] {
            let exec = write_exec_file(&[], &s_map, &[], 0);
            assert_eq!((exec[2], exec[3]), (0, 0));
            assert_eq!(exec.len(), EXEC_HEADER_WORDS + 3, "header plus an empty band section");
        }
    }

    /// The band section lands after the map, so a reader that stops at the map's end does not
    /// see it and one that knows about it finds it at the offset the header implies.
    #[test]
    fn write_exec_file_appends_bands_past_the_map() {
        use r1cs::types::{GateBand, GateBandKind};
        let adds = vec![[10u64, 20, 1, 1]];
        // Tall enough to contain both bands: a band's boundary is always inside the live extent.
        let s_map = vec![(1..=12).collect::<Vec<u32>>(), (13..=24).collect::<Vec<u32>>()];
        let bands = vec![
            GateBand { row: 0, kind: GateBandKind::Poseidon1CompressorCompression, payload: 0 },
            // A non-zero payload, so the triple stride is actually exercised: this is where BLAKE3
            // puts a block's `flags`, which the expander cannot read off the witness trace.
            GateBand { row: 10, kind: GateBandKind::Poseidon1CompressorSponge, payload: 0xB3 },
        ];

        let without = write_exec_file(&adds, &s_map, &[], 0);
        let with = write_exec_file(&adds, &s_map, &bands, 0);

        // 12 rows x 2 cols = 24 entries = 12 words.
        let prefix = EXEC_HEADER_WORDS + adds.len() * 4 + 12;
        assert_eq!(with[..prefix], without[..prefix], "the map must not move");
        assert_eq!(with[prefix], GATE_BAND_FORMAT_VERSION, "section version leads");
        assert_eq!(with[prefix + 1], 2, "band count");
        assert_eq!(with[prefix + 2], 0, "the per-air aux word");
        // three words per band: row, kind, payload
        assert_eq!(with[prefix + 3], 0);
        assert_eq!(with[prefix + 4], GateBandKind::Poseidon1CompressorCompression as u64);
        assert_eq!(with[prefix + 5], 0, "payload of the first band");
        assert_eq!(with[prefix + 6], 10);
        assert_eq!(with[prefix + 7], GateBandKind::Poseidon1CompressorSponge as u64);
        assert_eq!(with[prefix + 8], 0xB3, "payload of the second band");
        assert_eq!(with.len(), prefix + 3 + bands.len() * 3, "no slack past the last band");
        assert_eq!(without[prefix], GATE_BAND_FORMAT_VERSION);
        assert_eq!(without[prefix + 1], 0, "no bands");
    }

    /// Trimming must never drop a row a gate band needs: the prover would gather zeros for that
    /// band's input and the proof would fail a recursion layer later, far from the cause.
    #[test]
    #[should_panic(expected = "lies outside the map's live extent")]
    fn write_exec_file_refuses_a_band_outside_the_live_extent() {
        use r1cs::types::{GateBand, GateBandKind};
        let s_map = vec![vec![1u32, 2, 0, 0]]; // live extent is 2 rows
        let bands = vec![GateBand { row: 3, kind: GateBandKind::Poseidon1CompressorSponge, payload: 0 }];
        write_exec_file(&[], &s_map, &bands, 0);
    }

    #[test]
    fn test_invalid_setup_type() {
        let data = build_simple_r1cs_bytes();
        let options = PlonkOptions::default();
        let result = plonk2pil(&data, "invalid_type", &options);
        assert!(result.is_err());
    }
}
