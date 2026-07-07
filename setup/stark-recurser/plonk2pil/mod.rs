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
    /// Number of rows actually used before power-of-2 padding — mirrors JS `NUsed`.
    /// Used by the caller to compute the minimum nQueries for small recursive circuits (A2).
    pub n_used: usize,
    /// Airgroup name used in the PIL.
    pub airgroup_name: String,
    /// Air name used in the PIL.
    pub air_name: String,
}

/// Serialize PLONK additions and signal map into an exec buffer.
///
/// Layout (all u64 LE):
/// - [0]: number of additions
/// - [1]: number of rows per sMap column
/// - [2..2+adds*4]: additions (sl, sr, coef_l, coef_r) flattened
/// - [2+adds*4..]: sMap values interleaved by row (for each row: col0, col1, ...)
fn write_exec_file(adds: &[r1cs::to_plonk::PlonkAddition], s_map: &[Vec<u32>]) -> Vec<u64> {
    let n_adds = adds.len();
    let n_cols = s_map.len();
    let n_rows = if n_cols > 0 { s_map[0].len() } else { 0 };

    let size = 2 + n_adds * 4 + n_cols * n_rows;
    let mut buff = vec![0u64; size];

    buff[0] = n_adds as u64;
    buff[1] = n_rows as u64;

    for (i, add) in adds.iter().enumerate() {
        buff[2 + i * 4] = add[0];
        buff[2 + i * 4 + 1] = add[1];
        buff[2 + i * 4 + 2] = add[2];
        buff[2 + i * 4 + 3] = add[3];
    }

    let base = 2 + n_adds * 4;
    for i in 0..n_rows {
        for c in 0..n_cols {
            buff[base + n_cols * i + c] = s_map[c][i] as u64;
        }
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

    let exec = write_exec_file(&res.plonk_additions, &res.s_map);

    Ok(PlonkResult {
        exec,
        pil_str: res.pil_str,
        fixed_pols: res.fixed_pols,
        n_bits: res.n_bits,
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

    #[test]
    fn test_write_exec_file_roundtrip() {
        let adds: Vec<PlonkAddition> = vec![[10, 20, 30, 40], [50, 60, 70, 80]];
        let s_map: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];

        let exec = write_exec_file(&adds, &s_map);

        assert_eq!(exec[0], 2); // 2 additions
        assert_eq!(exec[1], 4); // 4 rows

        // First addition
        assert_eq!(exec[2], 10);
        assert_eq!(exec[3], 20);
        assert_eq!(exec[4], 30);
        assert_eq!(exec[5], 40);

        // Second addition
        assert_eq!(exec[6], 50);
        assert_eq!(exec[7], 60);
        assert_eq!(exec[8], 70);
        assert_eq!(exec[9], 80);

        // sMap data: row 0 => col0=1, col1=5
        let base = 2 + 2 * 4;
        assert_eq!(exec[base], 1);
        assert_eq!(exec[base + 1], 5);
        // row 1 => col0=2, col1=6
        assert_eq!(exec[base + 2], 2);
        assert_eq!(exec[base + 3], 6);
    }

    #[test]
    fn test_invalid_setup_type() {
        let data = build_simple_r1cs_bytes();
        let options = PlonkOptions::default();
        let result = plonk2pil(&data, "invalid_type", &options);
        assert!(result.is_err());
    }
}
