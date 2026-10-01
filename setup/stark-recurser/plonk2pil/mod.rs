//! plonk2pil: Convert R1CS constraint systems to PIL (Polynomial Identity Language).
//!
//! This module is the Rust port of the `recurser-js` pipeline.
//! It reads R1CS binary files, converts constraints to PLONK format, and runs
//! one of several setup routines to produce PIL source and fixed polynomials.
//!
//! The reader, the PLONK conversion and the `.exec` writer are generic over the r1cs's field
//! ([`field::PlonkField`]: Goldilocks or BN254). The setup families are over one field each: the
//! STARK recursion's (Poseidon1, Poseidon2, blake3) over Goldilocks, and the final SNARK wrap's
//! (PoseidonBN254, [`proofman_common::hash_family::BN254_WRAP_FAMILY`]) over BN254.
//!
//! The main entry point is [`plonk2pil`], which dispatches to the appropriate
//! setup variant based on the `setup_type` argument.

pub mod estimate;
pub mod field;
pub mod merge_copies;
pub mod packers;
pub mod r1cs;
pub mod setups;
pub mod utils;

// Re-export old flat module names as aliases for backward compatibility
pub use r1cs::to_plonk as r1cs2plonk;
pub use r1cs::types as r1cs_types;
pub use setups::poseidon1::aggregation as aggregation_setup;
pub use setups::poseidon1::compressor as compressor_setup;

use anyhow::{bail, Result};
use proofman_common::exec_format::{ExecLayout, GATE_BAND_FORMAT_VERSION, GATE_BAND_HEADER_WORDS, GATE_BAND_WORDS};
use proofman_common::hash_family::BN254_WRAP_FAMILY;
use proofman_fields::Bn254;

use field::{PlonkField, R1csPrime};
use r1cs::types::{r1cs_prime, read_r1cs_from_bytes, read_r1cs_header, PlonkOptions};
pub use r1cs::types::{FixedPol, SetupResult};

/// The result returned by [`plonk2pil`], containing everything needed
/// for downstream proof generation.
///
/// `V` is how its fixed columns hold a value, and so the field the r1cs is over ([`FixedValue`]):
/// the default, `u64`, the canonical word of a Goldilocks element, for the STARK recursion's
/// families; [`Bn254`] for the final SNARK wrap's.
#[derive(Debug, Clone)]
pub struct PlonkResult<V = u64> {
    /// Execution buffer: serialized additions and signal map.
    pub exec: Vec<u64>,
    /// Generated PIL source string.
    pub pil_str: String,
    /// Fixed polynomial values, as a flat list of (name, index, values).
    pub fixed_pols: Vec<FixedPol<V>>,
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

/// How a [`PlonkResult`] holds a fixed value, which names the field of the r1cs it is the result
/// of, and so the families that set it up: `u64` and [`Bn254`] (sealed).
pub trait FixedValue: sealed::FieldSetups {}

impl FixedValue for u64 {}
impl FixedValue for Bn254 {}

mod sealed {
    use anyhow::{bail, ensure, Result};
    use proofman_common::hash_family::BN254_WRAP_FAMILY;
    use proofman_fields::{Bn254, Goldilocks, PrimeField64};

    use super::field::PlonkField;
    use super::packers;
    use super::r1cs::types::{PlonkOptions, R1csFile, SetupResult};

    /// The families over a field, behind [`super::FixedValue`].
    pub trait FieldSetups: Sized {
        /// The field of the r1cs.
        type Field: PlonkField;

        /// The setup types of the families over [`Self::Field`].
        const SETUP_TYPES: &'static [&'static str];

        /// Runs the setup `setup_type` of the family `options.hash_id` on `r1cs`.
        fn setup(
            r1cs: &R1csFile<Self::Field>,
            setup_type: &str,
            options: &PlonkOptions,
        ) -> Result<SetupResult<Self::Field>>;

        /// A value of a fixed column, as the result holds it.
        fn from_field(value: Self::Field) -> Self;
    }

    /// The STARK recursion's families, over Goldilocks. Each value is its canonical word, as the
    /// STARK setup takes plonk2pil's columns and a `.const` file stores them.
    impl FieldSetups for u64 {
        type Field = Goldilocks;

        const SETUP_TYPES: &'static [&'static str] = &["compressor", "aggregation"];

        fn setup(
            r1cs: &R1csFile<Goldilocks>,
            setup_type: &str,
            options: &PlonkOptions,
        ) -> Result<SetupResult<Goldilocks>> {
            ensure!(
                options.hash_id != BN254_WRAP_FAMILY,
                "plonk2pil: the {BN254_WRAP_FAMILY} family sets up an r1cs over BN254, and this one is over Goldilocks"
            );
            packers::refuse_bn254_gates(r1cs)?;
            Ok(match setup_type {
                "compressor" => packers::pack_compressor(r1cs, options),
                "aggregation" => packers::pack_aggregation(r1cs, options),
                other => bail!("Invalid setup type: '{other}'. Must be one of: {}", Self::SETUP_TYPES.join(", ")),
            })
        }

        fn from_field(value: Goldilocks) -> u64 {
            value.as_canonical_u64()
        }
    }

    /// The final SNARK wrap's family, over BN254, whose fixed columns go to the pilfflonk setup as
    /// they are.
    impl FieldSetups for Bn254 {
        type Field = Bn254;

        const SETUP_TYPES: &'static [&'static str] = &["wrap"];

        fn setup(r1cs: &R1csFile<Bn254>, setup_type: &str, options: &PlonkOptions) -> Result<SetupResult<Bn254>> {
            match setup_type {
                "wrap" => packers::pack_wrap(r1cs, options),
                other => bail!("Invalid setup type: '{other}'. Must be one of: {}", Self::SETUP_TYPES.join(", ")),
            }
        }

        fn from_field(value: Bn254) -> Bn254 {
            value
        }
    }
}

/// Serialize PLONK additions, the signal map and the gate bands into an exec buffer, in the layout
/// of [`proofman_common::exec_format`] and the version `F` is written in: version 2 over
/// Goldilocks, the one the STARK prover reads, and version 3, which records the coefficient width,
/// over BN254.
///
/// The map is stored at its live extent, not the trace's: the packers fill rows from 0 and leave
/// the power-of-two padding untouched, and the columns a gate band fills are never mapped -- the
/// expander writes those from the band's boundary. `getCommitedPols` zeroes everything outside the
/// extent, which is what those cells held anyway. Both bounds are measured rather than assumed, so
/// a packer that starts using a row or column cannot silently have it dropped.
///
/// Public so that a caller can write the exec of rows it places itself, as the tests of the pilfflonk
/// wrap's witness (`pilfflonk-wrap-witness`) do over BN254.
pub fn write_exec_file<F: PlonkField>(
    adds: &[r1cs::to_plonk::PlonkAddition<F>],
    s_map: &[Vec<u32>],
    gate_bands: &[r1cs::types::GateBand],
    band_aux: u64,
) -> Vec<u64> {
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

    let layout = ExecLayout::new::<F>(adds.len(), map_rows, map_cols);
    let map_at = layout.map_at();
    let bands_at = layout.bands_at();
    let mut buff = vec![0u64; bands_at + GATE_BAND_HEADER_WORDS + gate_bands.len() * GATE_BAND_WORDS];

    layout.write_header(&mut buff);

    let addition_words = buff[layout.header_words()..map_at].chunks_exact_mut(layout.addition_words());
    for (add, words) in adds.iter().zip(addition_words) {
        let (wires, coeffs) = words.split_at_mut(2);
        wires[0] = add.wires[0] as u64;
        wires[1] = add.wires[1] as u64;
        let (coef_l, coef_r) = coeffs.split_at_mut(F::COEF_WORDS);
        add.coeffs[0].write_exec_words(coef_l);
        add.coeffs[1].write_exec_words(coef_r);
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

    buff[bands_at..bands_at + GATE_BAND_HEADER_WORDS].copy_from_slice(&[
        GATE_BAND_FORMAT_VERSION,
        gate_bands.len() as u64,
        band_aux,
    ]);
    let band_words = buff[bands_at + GATE_BAND_HEADER_WORDS..].chunks_exact_mut(GATE_BAND_WORDS);
    for (b, words) in gate_bands.iter().zip(band_words) {
        words.copy_from_slice(&[b.row as u64, b.kind as u64, b.payload]);
    }

    buff
}

/// Read an R1CS file and run the specified setup to produce PIL and fixed polynomials.
///
/// # Arguments
/// * `r1cs_data` - Raw bytes of the R1CS binary file.
/// * `setup_type` - One of `"compressor"`, `"aggregation"` over Goldilocks, `"wrap"` over BN254.
/// * `options` - Optional configuration (airgroup name, max constraint degree, the family).
///
/// # Returns
/// A [`PlonkResult`] containing the exec buffer, PIL source, and fixed polynomials.
///
/// The result's type says the field: a [`PlonkResult`] (`u64` values) is the STARK recursion's,
/// over Goldilocks, and a `PlonkResult<Bn254>` the final SNARK wrap's, over BN254, as
/// [`read_r1cs_from_bytes`] reads an r1cs into the field it is asked for. An r1cs over the other
/// field is refused, saying which families set it up, and one over any other prime as unknown; so
/// is a family that is not over the field.
pub fn plonk2pil<V: FixedValue>(r1cs_data: &[u8], setup_type: &str, options: &PlonkOptions) -> Result<PlonkResult<V>> {
    if !V::SETUP_TYPES.contains(&setup_type) {
        bail!("Invalid setup type: '{}'. Must be one of: {}", setup_type, V::SETUP_TYPES.join(", "));
    }

    match r1cs_prime(&read_r1cs_header(r1cs_data)?)? {
        prime if prime == V::Field::PRIME => {}
        R1csPrime::Bn254 => bail!(
            "plonk2pil: an r1cs over BN254 is set up by the {BN254_WRAP_FAMILY} family, into a PlonkResult<Bn254>, \
             and not by the STARK recursion's families, which are over Goldilocks"
        ),
        R1csPrime::Goldilocks => bail!(
            "plonk2pil: an r1cs over Goldilocks is set up by the STARK recursion's families, into a PlonkResult, \
             and not by the {BN254_WRAP_FAMILY} family, which is over BN254"
        ),
    }
    let r1cs = read_r1cs_from_bytes::<V::Field>(r1cs_data)?;
    let res = V::setup(&r1cs, setup_type, options)?;

    let exec = write_exec_file(&res.plonk_additions, &res.s_map, &res.gate_bands, res.band_aux);

    Ok(PlonkResult {
        exec,
        pil_str: res.pil_str,
        fixed_pols: res.fixed_pols.into_iter().map(fixed_values).collect(),
        n_bits: res.n_bits,
        n_bits_natural: res.n_bits_natural,
        n_used: res.n_used,
        airgroup_name: res.airgroup_name,
        air_name: res.air_name,
    })
}

/// A fixed column as the result holds it.
fn fixed_values<V: FixedValue>(pol: FixedPol<V::Field>) -> FixedPol<V> {
    // `Goldilocks` and `u64` share a layout, as `Bn254` does with itself, so the collect reuses
    // the column's allocation.
    FixedPol { name: pol.name, index: pol.index, values: pol.values.into_iter().map(V::from_field).collect() }
}

#[cfg(test)]
mod tests {
    use super::r1cs::to_plonk::*;
    use super::r1cs::types::read_r1cs_from_bytes;
    use super::*;
    use proofman_common::exec_format::{
        ExecFile, ExecGateBand, EXEC_FORMAT_VERSION, EXEC_FORMAT_VERSION_WIDE, EXEC_HEADER_WORDS, EXEC_MAGIC,
        EXEC_WIDE_HEADER_WORDS,
    };
    use proofman_fields::{Field, Goldilocks, PrimeField, QuotientMap};

    /// Run the real compressor packer end-to-end on an r1cs (exercises the row-count
    /// assert + verify_merge_soundness). ESTIMATE_HASH picks the family; it defaults to
    /// `hash_family::DEFAULT_HASH_ID`.
    ///   ESTIMATE_R1CS=/path/x.r1cs [ESTIMATE_HASH=Poseidon2] \
    ///     cargo test -p pil2-stark-recurser run_compressor --release -- --ignored --nocapture
    #[test]
    #[ignore]
    fn run_compressor() {
        use proofman_common::hash_family::GateRole;
        let Ok(f) = std::env::var("ESTIMATE_R1CS") else {
            eprintln!("set ESTIMATE_R1CS=/path/to/file.r1cs");
            return;
        };
        let bytes = std::fs::read(&f).unwrap_or_else(|e| panic!("read {f}: {e}"));
        let hash_id =
            std::env::var("ESTIMATE_HASH").unwrap_or_else(|_| proofman_common::hash_family::DEFAULT_HASH_ID.into());
        let opts = PlonkOptions {
            airgroup_name: Some("Compressor".into()),
            max_constraint_degree: Some(5),
            hash_id,
            merge_copies: true,
            blake3_lanes: None,
            min_n_bits: None,
        };
        let res: PlonkResult = plonk2pil(&bytes, "compressor", &opts).expect("compressor packing failed");
        let r1cs = read_r1cs_from_bytes::<Goldilocks>(&bytes).unwrap();
        let cgi = get_custom_gates_info(&r1cs);
        let n_pos = cgi.n(GateRole::PoseidonCompression) + cgi.n(GateRole::PoseidonSponge);
        eprintln!("\n=== {f}  compressor OK: nBits={} nUsed={} n_pos={}", res.n_bits, res.n_used, n_pos);
    }

    /// Build a minimal Goldilocks R1CS with a single multiplication constraint and no custom gates.
    fn build_simple_r1cs_bytes() -> Vec<u8> {
        r1cs_bytes(&R1csPrime::Goldilocks.modulus_le(), &one(8))
    }

    /// `1`, little-endian in `n8` bytes.
    fn one(n8: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; n8];
        bytes[0] = 1;
        bytes
    }

    /// An R1CS over `prime` (little-endian, its length the `n8`) with the single constraint
    /// `wire_1 * wire_2 = coeff * wire_3`.
    fn r1cs_bytes(prime: &[u8], coeff: &[u8]) -> Vec<u8> {
        let n8 = prime.len();
        assert_eq!(coeff.len(), n8);
        let mut buf: Vec<u8> = Vec::new();

        buf.extend_from_slice(b"r1cs");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // 2 sections

        // Header
        let mut hdr: Vec<u8> = Vec::new();
        hdr.extend_from_slice(&(n8 as u32).to_le_bytes());
        hdr.extend_from_slice(prime);
        hdr.extend_from_slice(&4u32.to_le_bytes()); // nVars
        hdr.extend_from_slice(&1u32.to_le_bytes()); // nOutputs
        hdr.extend_from_slice(&1u32.to_le_bytes()); // nPubInputs
        hdr.extend_from_slice(&1u32.to_le_bytes()); // nPrvInputs
        hdr.extend_from_slice(&4u64.to_le_bytes()); // nLabels
        hdr.extend_from_slice(&1u32.to_le_bytes()); // nConstraints

        buf.extend_from_slice(&1u32.to_le_bytes()); // Section type 1 = header
        buf.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        buf.extend_from_slice(&hdr);

        // Constraint: wire_1 * wire_2 = coeff * wire_3
        let mut cdata: Vec<u8> = Vec::new();
        // A: 1 term (wire=1, coeff=1)
        cdata.extend_from_slice(&1u32.to_le_bytes());
        cdata.extend_from_slice(&1u32.to_le_bytes());
        cdata.extend_from_slice(&one(n8));
        // B: 1 term (wire=2, coeff=1)
        cdata.extend_from_slice(&1u32.to_le_bytes());
        cdata.extend_from_slice(&2u32.to_le_bytes());
        cdata.extend_from_slice(&one(n8));
        // C: 1 term (wire=3, coeff)
        cdata.extend_from_slice(&1u32.to_le_bytes());
        cdata.extend_from_slice(&3u32.to_le_bytes());
        cdata.extend_from_slice(coeff);

        buf.extend_from_slice(&2u32.to_le_bytes()); // Section type 2 = constraints
        buf.extend_from_slice(&(cdata.len() as u64).to_le_bytes());
        buf.extend_from_slice(&cdata);

        buf
    }

    #[test]
    fn test_r1cs2plonk_basic() {
        let data = build_simple_r1cs_bytes();
        let r1cs = read_r1cs_from_bytes::<Goldilocks>(&data).unwrap();
        let (constraints, additions) = r1cs2plonk(&r1cs);

        assert_eq!(constraints.len(), 1);
        assert!(additions.is_empty());
        // Should be a multiplication gate: qM != 0
        assert!(!constraints[0].coeffs[0].is_zero());
    }

    /// The reader takes an element of the file's own width: a BN254 coefficient wider than any
    /// u64 comes through whole.
    #[test]
    fn a_bn254_r1cs_is_read_at_32_bytes() {
        use num_bigint::BigUint;
        let r = R1csPrime::Bn254.modulus_le();
        let mut minus_two = (BigUint::from_bytes_le(&r) - 2u32).to_bytes_le();
        minus_two.resize(32, 0);
        let r1cs = read_r1cs_from_bytes::<Bn254>(&r1cs_bytes(&r, &minus_two)).unwrap();
        assert_eq!(r1cs.header.n8, 32);
        assert_eq!(r1cs.constraints[0].c[&3], -Bn254::TWO);
        let (constraints, _) = r1cs2plonk(&r1cs);
        assert_eq!(constraints[0].coeffs[3], Bn254::TWO, "qO = -c");
    }

    /// Reading a file into a field it is not over is an error naming both, never a misread.
    #[test]
    fn an_r1cs_is_refused_in_a_field_it_is_not_over() {
        let gl = build_simple_r1cs_bytes();
        let err = read_r1cs_from_bytes::<Bn254>(&gl).unwrap_err().to_string();
        assert!(err.contains("over Goldilocks") && err.contains("BN254"), "{err}");

        let r = R1csPrime::Bn254.modulus_le();
        let err = read_r1cs_from_bytes::<Goldilocks>(&r1cs_bytes(&r, &one(32))).unwrap_err().to_string();
        assert!(err.contains("over BN254") && err.contains("Goldilocks"), "{err}");
    }

    #[test]
    fn an_unknown_prime_is_refused_naming_it() {
        let seven = 7u64.to_le_bytes();
        let err = read_r1cs_header(&r1cs_bytes(&seven, &one(8))).and_then(|h| r1cs_prime(&h)).unwrap_err().to_string();
        assert!(err.contains("prime 7 (n8 = 8)"), "{err}");
        let err = plonk2pil::<u64>(&r1cs_bytes(&seven, &one(8)), "compressor", &PlonkOptions::default()).unwrap_err();
        assert!(err.to_string().contains("prime 7"), "{err}");
    }

    /// A coefficient at or above the prime is not an element: refused, not reduced.
    #[test]
    fn a_non_canonical_coefficient_is_refused() {
        let p = R1csPrime::Goldilocks.modulus_le();
        let err = read_r1cs_from_bytes::<Goldilocks>(&r1cs_bytes(&p, &p)).unwrap_err().to_string();
        assert!(err.contains("18446744069414584321 is not an element of Goldilocks"), "{err}");
    }

    /// A truncated file is an error at the read that runs out, whatever length it claims.
    #[test]
    fn a_truncated_r1cs_is_an_error() {
        let data = build_simple_r1cs_bytes();
        for len in [0, 3, 12, 30, data.len() - 1] {
            assert!(read_r1cs_from_bytes::<Goldilocks>(&data[..len]).is_err(), "{len} bytes");
        }
    }

    /// The STARK recursion's families are over Goldilocks: asked for a `PlonkResult` of words,
    /// plonk2pil refuses a BN254 r1cs up front, naming the family that sets it up, for both setup
    /// types and whichever Goldilocks family.
    #[test]
    fn the_goldilocks_families_refuse_a_bn254_r1cs() {
        let bn254 = r1cs_bytes(&R1csPrime::Bn254.modulus_le(), &one(32));
        for setup_type in ["compressor", "aggregation"] {
            for hash_id in proofman_common::hash_family::FAMILIES {
                let opts = PlonkOptions { hash_id: hash_id.to_string(), ..Default::default() };
                let err = plonk2pil::<u64>(&bn254, setup_type, &opts).unwrap_err().to_string();
                assert!(err.contains("an r1cs over BN254 is set up by the PoseidonBN254 family"), "{err}");
            }
        }
    }

    /// The wrap's family is over BN254: it refuses a Goldilocks r1cs, and the other families refuse
    /// to set up a BN254 one into a `PlonkResult<Bn254>`.
    #[test]
    fn the_wrap_family_and_the_bn254_field_go_together() {
        let goldilocks = build_simple_r1cs_bytes();
        let wrap = PlonkOptions { hash_id: BN254_WRAP_FAMILY.to_string(), ..Default::default() };
        let err = plonk2pil::<Bn254>(&goldilocks, "wrap", &wrap).unwrap_err().to_string();
        assert!(err.contains("an r1cs over Goldilocks is set up by the STARK recursion's families"), "{err}");
        let err = plonk2pil::<u64>(&goldilocks, "compressor", &wrap).unwrap_err().to_string();
        assert!(err.contains("the PoseidonBN254 family sets up an r1cs over BN254"), "{err}");

        let bn254 = r1cs_bytes(&R1csPrime::Bn254.modulus_le(), &one(32));
        for hash_id in proofman_common::hash_family::FAMILIES {
            let opts = PlonkOptions { hash_id: hash_id.to_string(), ..Default::default() };
            let err = plonk2pil::<Bn254>(&bn254, "wrap", &opts).unwrap_err().to_string();
            assert!(err.contains(&format!("the {hash_id} family is over Goldilocks")), "{err}");
        }
        let err = plonk2pil::<Bn254>(&bn254, "compressor", &wrap).unwrap_err().to_string();
        assert!(err.contains("Must be one of: wrap"), "{err}");
        let err = plonk2pil::<u64>(&goldilocks, "wrap", &PlonkOptions::default()).unwrap_err().to_string();
        assert!(err.contains("Must be one of: compressor, aggregation"), "{err}");

        let res = plonk2pil::<Bn254>(&bn254, "wrap", &wrap).unwrap();
        let exec = ExecFile::<Bn254>::from_words(&res.exec).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(exec.layout.version(), EXEC_FORMAT_VERSION_WIDE, "a BN254 exec is version 3");
        assert!(res.pil_str.contains("require \"poseidon_bn254/wrap.pil\";"), "{}", res.pil_str);
    }

    /// A Goldilocks r1cs with a gate of the BN254 wrap is refused: the STARK families would leave
    /// it unplaced.
    #[test]
    fn a_goldilocks_r1cs_with_a_wrap_gate_is_refused() {
        let mut r1cs = read_r1cs_from_bytes::<Goldilocks>(&build_simple_r1cs_bytes()).unwrap();
        r1cs.custom_gates.push(r1cs::types::CustomGate { template_name: "PoseidonT".into(), parameters: vec![] });
        let err = packers::refuse_bn254_gates(&r1cs).unwrap_err().to_string();
        assert!(err.contains("uses PoseidonT, a gate of the PoseidonBN254 family"), "{err}");
    }

    /// One map entry, by row and column, out of the packed u32 pairs.
    fn map_entry(exec: &[u64], map_at: usize, map_cols: usize, row: usize, col: usize) -> u32 {
        let entry = row * map_cols + col;
        (exec[map_at + entry / 2] >> (32 * (entry % 2))) as u32
    }

    #[test]
    fn test_write_exec_file_roundtrip() {
        let adds: Vec<PlonkAddition<Goldilocks>> = vec![
            PlonkAddition { wires: [10, 20], coeffs: [Goldilocks::new(30), Goldilocks::new(40)] },
            PlonkAddition { wires: [50, 60], coeffs: [Goldilocks::new(70), Goldilocks::new(80)] },
        ];
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

        let exec = write_exec_file::<Goldilocks>(&[], &s_map, &[], 0);

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
        let exec = write_exec_file::<Goldilocks>(&[], &s_map, &[], 0);

        assert_eq!((exec[2], exec[3]), (1, 1));
        assert_eq!(exec[EXEC_HEADER_WORDS], 7, "the entry, with the high half unused");
        assert_eq!(exec[EXEC_HEADER_WORDS + 1], GATE_BAND_FORMAT_VERSION, "section starts a word later");
        assert_eq!(exec.len(), EXEC_HEADER_WORDS + 4, "version, count and aux");
    }

    /// An all-zero map has no live extent at all and must not produce a negative or wrapped size.
    #[test]
    fn write_exec_file_handles_an_empty_map() {
        for s_map in [vec![], vec![vec![0u32; 4]; 3]] {
            let exec = write_exec_file::<Goldilocks>(&[], &s_map, &[], 0);
            assert_eq!((exec[2], exec[3]), (0, 0));
            assert_eq!(exec.len(), EXEC_HEADER_WORDS + 3, "header plus an empty band section");
        }
    }

    /// The band section lands after the map, so a reader that stops at the map's end does not
    /// see it and one that knows about it finds it at the offset the header implies.
    #[test]
    fn write_exec_file_appends_bands_past_the_map() {
        use r1cs::types::{GateBand, GateBandKind};
        let adds = vec![PlonkAddition { wires: [10, 20], coeffs: [Goldilocks::ONE; 2] }];
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
        write_exec_file::<Goldilocks>(&[], &s_map, &bands, 0);
    }

    /// The map and bands the round trips below write in each field: 5 columns x 6 rows of trace,
    /// of which 3 x 3 are live (9 entries, so the map's last word is half empty), and two bands with
    /// a payload and an aux word, as BLAKE3 writes them.
    fn exec_fixture() -> (Vec<Vec<u32>>, Vec<r1cs::types::GateBand>, u64) {
        use r1cs::types::{GateBand, GateBandKind};
        let s_map =
            vec![vec![1, 2, 3, 0, 0, 0], vec![4, 0, 6, 0, 0, 0], vec![0, 8, 9, 0, 0, 0], vec![0; 6], vec![0; 6]];
        let bands = vec![
            GateBand { row: 0, kind: GateBandKind::Blake3Node, payload: 0xB3 },
            GateBand { row: 2, kind: GateBandKind::Blake3CompressChunk, payload: 0 },
        ];
        (s_map, bands, 0x41_0000_0004)
    }

    /// Writes `adds` over `F` with the fixture's map and bands and reads the buffer back: the
    /// additions come back whole, the map trimmed to its live extent and reading as the trace's
    /// everywhere, and the bands as they were.
    fn assert_exec_round_trips<F: PlonkField>(adds: &[PlonkAddition<F>]) {
        let (s_map, bands, aux) = exec_fixture();
        let exec = write_exec_file(adds, &s_map, &bands, aux);
        let file = ExecFile::<F>::from_words(&exec).unwrap_or_else(|e| panic!("{e}"));

        assert_eq!((file.layout.version(), file.layout.coef_words()), (F::EXEC_VERSION, F::COEF_WORDS));
        assert_eq!((file.layout.map_rows(), file.layout.map_cols()), (3, 3), "trimmed to the live extent");
        let read: Vec<_> = file.additions.iter().map(|a| (a.wires, a.coeffs)).collect();
        let written: Vec<_> = adds.iter().map(|a| (a.wires, a.coeffs)).collect();
        assert_eq!(read, written);
        for (col, column) in s_map.iter().enumerate() {
            for (row, &signal) in column.iter().enumerate() {
                assert_eq!(file.map_entry(row, col), signal, "row {row} col {col}");
            }
        }
        assert_eq!(file.band_aux, aux);
        let written: Vec<_> =
            bands.iter().map(|b| ExecGateBand { row: b.row as u64, kind: b.kind as u64, payload: b.payload }).collect();
        assert_eq!(file.bands, written);
    }

    #[test]
    fn a_goldilocks_exec_reads_back_as_written() {
        assert_exec_round_trips(&[
            PlonkAddition { wires: [10, 20], coeffs: [Goldilocks::new(30), Goldilocks::NEG_ONE] },
            PlonkAddition { wires: [0, u32::MAX], coeffs: [Goldilocks::ZERO, Goldilocks::new(1 << 63)] },
        ]);
    }

    /// Coefficients no word holds come back whole.
    #[test]
    fn a_bn254_exec_reads_back_as_written() {
        let adds = [
            PlonkAddition { wires: [10, 20], coeffs: [Bn254::NEG_ONE, Bn254::from_int(1u128 << 100)] },
            PlonkAddition { wires: [0, u32::MAX], coeffs: [Bn254::ZERO, -Bn254::TWO] },
        ];
        assert!(adds.iter().flat_map(|a| a.coeffs).all(|c| c.is_zero() || c.as_canonical_biguint().bits() > 64));
        assert_exec_round_trips(&adds);
    }

    /// Over BN254 the header records four words per coefficient, and a coefficient is its canonical
    /// value in them, least significant first: the original pil-fflonk's 32-byte `Fr`, in words.
    #[test]
    fn a_bn254_exec_is_version_3_with_four_word_coefficients() {
        let adds = [PlonkAddition { wires: [7, 9], coeffs: [Bn254::NEG_ONE, Bn254::from_int(1u128 << 100)] }];
        let exec = write_exec_file(&adds, &[vec![1, 2]], &[], 0);

        assert_eq!(exec[..EXEC_WIDE_HEADER_WORDS], [EXEC_MAGIC | EXEC_FORMAT_VERSION_WIDE, 1, 2, 1, 4]);
        assert_eq!(exec[5..7], [7, 9], "the wires, a word each");
        let r_minus_1 = [0x43e1f593f0000000, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029];
        assert_eq!(exec[7..11], r_minus_1, "coef_l = r - 1");
        assert_eq!(exec[11..15], [0, 1 << 36, 0, 0], "coef_r = 2^100");
        assert_eq!(exec[15], 1 | (2 << 32), "the 2 x 1 map");
        assert_eq!(exec[16..], [GATE_BAND_FORMAT_VERSION, 0, 0], "an empty band section");
    }

    /// Past the additions both versions are the same words: the map and the bands do not depend on
    /// the field.
    #[test]
    fn the_map_and_the_bands_are_the_same_words_in_both_versions() {
        let (s_map, bands, aux) = exec_fixture();
        let gl = write_exec_file(&[PlonkAddition { wires: [1, 2], coeffs: [Goldilocks::ONE; 2] }], &s_map, &bands, aux);
        let bn = write_exec_file(&[PlonkAddition { wires: [1, 2], coeffs: [Bn254::ONE; 2] }], &s_map, &bands, aux);
        assert_eq!(gl[1..EXEC_HEADER_WORDS], bn[1..EXEC_HEADER_WORDS], "the extents");
        assert_eq!(gl[EXEC_HEADER_WORDS + 4..], bn[EXEC_WIDE_HEADER_WORDS + 10..]);
    }

    /// No prefix of a written exec reads, nor does one with a word too many: the reader never runs
    /// off a buffer, nor ignores what it does not account for.
    #[test]
    fn a_truncated_or_padded_exec_is_refused() {
        let (s_map, bands, aux) = exec_fixture();
        let exec =
            write_exec_file(&[PlonkAddition { wires: [1, 2], coeffs: [Bn254::NEG_ONE; 2] }], &s_map, &bands, aux);
        for len in 0..exec.len() {
            assert!(ExecFile::<Bn254>::from_words(&exec[..len]).is_err(), "{len} of {} words", exec.len());
        }
        let mut padded = exec;
        padded.push(0);
        assert!(ExecFile::<Bn254>::from_words(&padded).is_err());
    }

    /// An exec reads only in the field it was written over.
    #[test]
    fn an_exec_is_refused_in_a_field_it_is_not_over() {
        let s_map = [vec![1u32]];
        let gl = write_exec_file(&[PlonkAddition { wires: [1, 2], coeffs: [Goldilocks::ONE; 2] }], &s_map, &[], 0);
        let bn = write_exec_file(&[PlonkAddition { wires: [1, 2], coeffs: [Bn254::ONE; 2] }], &s_map, &[], 0);
        let err = ExecFile::<Bn254>::from_words(&gl).unwrap_err().to_string();
        assert!(err.contains("version 2 with 1-word coefficients, not the 4-word ones of BN254"), "{err}");
        let err = ExecFile::<Goldilocks>::from_words(&bn).unwrap_err().to_string();
        assert!(err.contains("version 3 with 4-word coefficients, not the 1-word ones of Goldilocks"), "{err}");
    }

    /// The STARK prover's loader still takes the Goldilocks exec as it is, and refuses the BN254 one
    /// as what it is rather than as a key from another build. Both read back from disk with
    /// [`ExecFile::read`].
    #[test]
    fn the_stark_loader_refuses_a_bn254_exec_by_name() {
        let s_map = [vec![1u32, 2]];
        let gl = write_exec_file(&[PlonkAddition { wires: [1, 2], coeffs: [Goldilocks::ONE; 2] }], &s_map, &[], 0);
        let bn = write_exec_file(&[PlonkAddition { wires: [1, 2], coeffs: [Bn254::NEG_ONE; 2] }], &s_map, &[], 0);
        let path = |name: &str| std::env::temp_dir().join(format!("plonk2pil_exec_{name}_{}.exec", std::process::id()));
        let (gl_path, bn_path) = (path("goldilocks"), path("bn254"));
        let bytes = |words: &[u64]| words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
        std::fs::write(&gl_path, bytes(&gl)).unwrap();
        std::fs::write(&bn_path, bytes(&bn)).unwrap();

        let loaded = proofman_common::load_exec_file(gl_path.to_str().unwrap(), 1);
        let refused = proofman_common::load_exec_file(bn_path.to_str().unwrap(), 1);
        let read_gl = ExecFile::<Goldilocks>::read(&gl_path);
        let read_bn = ExecFile::<Bn254>::read(&bn_path);
        // Best effort: a leftover file in the temporary directory harms nothing.
        let _ = std::fs::remove_file(&gl_path);
        let _ = std::fs::remove_file(&bn_path);

        assert_eq!(loaded.unwrap_or_else(|e| panic!("{e}")), gl);
        let err = refused.unwrap_err().to_string();
        assert!(
            err.contains(
                "is format version 3, the BN254 exec plonk2pil writes for the pilfflonk wrap; the STARK \
                          prover reads only version 2"
            ),
            "{err}"
        );
        assert_eq!(read_gl.unwrap_or_else(|e| panic!("{e}")), ExecFile::from_words(&gl).unwrap());
        assert_eq!(read_bn.unwrap_or_else(|e| panic!("{e}")), ExecFile::from_words(&bn).unwrap());
    }

    #[test]
    fn test_invalid_setup_type() {
        let data = build_simple_r1cs_bytes();
        let options = PlonkOptions::default();
        let result = plonk2pil::<u64>(&data, "invalid_type", &options);
        assert!(result.is_err());
    }
}
