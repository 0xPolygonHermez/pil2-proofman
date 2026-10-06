//! R1CS types and direct `&[u8]` parser for use by plonk2pil setup routines.
//!
//! An r1cs is read into the field it is over (`F`, a [`PlonkField`]): its coefficients and its
//! custom gates' parameters become elements of `F`, its wire ids and signals stay integers.
//! No temporary files.

use std::collections::{BTreeMap, HashMap};
use std::io::{Cursor, Read, Seek, SeekFrom};

use anyhow::{anyhow, bail, ensure, Result};
use num_bigint::BigUint;

use crate::plonk2pil::field::{PlonkField, R1csPrime};
use crate::plonk2pil::r1cs::to_plonk::PlonkAddition;

// ─── Public types ───────────────────────────────────────────────────────────

/// Wire index → coefficient. Ordered, so every pass over a combination's terms sees them in the
/// same order and the plonk placement it feeds is reproducible.
pub type LinearCombination<F> = BTreeMap<u32, F>;

/// A single R1CS constraint: A * B = C.
///
/// Ordered by the canonical values of its coefficients (the order of `F`), which is what the reader
/// sorts the constraints by and so what the row placement follows.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct R1csConstraint<F> {
    pub a: LinearCombination<F>,
    pub b: LinearCombination<F>,
    pub c: LinearCombination<F>,
}

/// R1CS header fields.
#[derive(Debug, Clone)]
pub struct R1csHeader {
    pub n8: u32,
    pub prime_bytes: Vec<u8>,
    pub n_vars: u32,
    pub n_outputs: u32,
    pub n_pub_inputs: u32,
    pub n_prv_inputs: u32,
    pub n_labels: u64,
    pub n_constraints: u32,
    pub use_custom_gates: bool,
}

/// A custom gate definition (template name + parameters).
#[derive(Debug, Clone)]
pub struct CustomGate<F> {
    pub template_name: String,
    pub parameters: Vec<F>,
}

/// One application of a custom gate (gate index + wire signals).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CustomGateUse {
    pub id: u32,
    pub signals: Vec<u64>,
}

/// Complete parsed R1CS file, over the field `F`.
#[derive(Debug, Clone)]
pub struct R1csFile<F> {
    pub header: R1csHeader,
    pub constraints: Vec<R1csConstraint<F>>,
    pub wire_to_label: Vec<u64>,
    pub custom_gates: Vec<CustomGate<F>>,
    pub custom_gates_uses: Vec<CustomGateUse>,
}

// ─── Setup result types (shared by all four setup variants) ─────────────────

#[derive(Debug, Clone)]
pub struct PlonkOptions {
    pub airgroup_name: Option<String>,
    pub max_constraint_degree: Option<usize>,
    pub hash_id: String,
    /// Eliminate exact copy constraints by merging signals; the connection argument
    /// then enforces the equality. Soundness depends on the s_map remap sweep being
    /// applied to custom-gate I/O cells (see [`crate::plonk2pil::merge_copies`]).
    pub merge_copies: bool,
    /// Parallel BLAKE3 permutations per 56-row block. Only the blake3 family reads this; the air
    /// caps it at 8 (above that the boundary opening depth exceeds 7). Defaults to 4, the point
    /// spec 5.3 settles on: 37,444 permutations of capacity at N=2^19.
    pub blake3_lanes: Option<usize>,
    /// Floor for the air's `nBits`. A recursive air that reuses another air's starkSetup must
    /// compile to the SAME number of rows as that setup describes -- the starkinfo is what the
    /// const file, the witness and the proof are all read against. Without a floor each air sizes
    /// itself to its own gate count, and two airs of an airgroup that land in different
    /// power-of-two buckets produce a const file the reused starkinfo cannot describe.
    pub min_n_bits: Option<usize>,
}

impl Default for PlonkOptions {
    fn default() -> Self {
        Self {
            airgroup_name: None,
            max_constraint_degree: None,
            hash_id: proofman_common::hash_family::DEFAULT_HASH_ID.to_string(),
            merge_copies: true,
            blake3_lanes: None,
            min_n_bits: None,
        }
    }
}

/// A single fixed polynomial column.
///
/// `V` is a field element while plonk2pil builds the column. The default, `u64`, is the canonical
/// word of a Goldilocks value, which is how [`crate::plonk2pil::PlonkResult`] hands the columns to
/// the STARK setup and how a `.const` file stores them.
#[derive(Debug, Clone)]
pub struct FixedPol<V = u64> {
    pub name: String,
    pub index: usize,
    pub values: Vec<V>,
}

/// One expandable row band: where it starts, and which gate shape fills it. The layout
/// within a band is fixed per shape, so nothing else needs recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateBand {
    pub row: u32,
    pub kind: GateBandKind,
    /// A per-block constant the expander needs but cannot read off the witness trace.
    ///
    /// BLAKE3 puts `flags` here -- st[15], a compile-time constant of the circom gate that the AIR
    /// carries as a FIXED column, so it is in neither the witness nor anything `expand_gate_bands`
    /// is handed. The BN128 wrap's range check puts its number of chunks. The poseidon kinds need
    /// nothing and write 0.
    pub payload: u64,
}

/// Gate shapes with recomputable interiors, and the BN128 wrap's range-check rows. Serialized into
/// the exec file, so the discriminants are a wire format -- append, never renumber.
///
/// Both axes are encoded because an expander needs both: the setup type fixes the band geometry
/// (10 rows with one chain slot, or 5 with two) and the family picks the permutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum GateBandKind {
    Poseidon1CompressorSponge = 1,
    Poseidon1CompressorCompression = 2,
    Poseidon1AggregationSponge = 3,
    Poseidon1AggregationCompression = 4,
    Poseidon2CompressorSponge = 5,
    Poseidon2CompressorCompression = 6,
    Poseidon2AggregationSponge = 7,
    Poseidon2AggregationCompression = 8,
    /// 56-row block hosting LANES permutations; the 65 columns per lane are the expander's.
    Blake3Node = 9,
    Blake3CompressChunk = 10,
    Blake3CompressParent = 11,
    /// A `Num2Bytes` row of the BN128 wrap: nothing to rebuild, the map gathers it whole, and the
    /// wrap's witness counts its chunks into the multiplicity column
    /// ([`RANGE_CHECK_BAND_KIND`](proofman_common::exec_format::RANGE_CHECK_BAND_KIND)).
    PoseidonBn128WrapRangeCheck = proofman_common::exec_format::RANGE_CHECK_BAND_KIND,
    /// A range-check row of the blake3 BN128 wrap: three `Num2Bytes` uses of `payload` chunks,
    /// which its witness counts into blake3's 16-bit table
    /// ([`BLAKE3_WRAP_RANGE_CHECK_BAND_KIND`](proofman_common::exec_format::BLAKE3_WRAP_RANGE_CHECK_BAND_KIND)).
    Blake3Bn128WrapRangeCheck = proofman_common::exec_format::BLAKE3_WRAP_RANGE_CHECK_BAND_KIND,
    /// A block of the blake3 BN128 wrap, of each kind, `payload` its flags.
    Blake3Bn128WrapNode = proofman_common::exec_format::BLAKE3_WRAP_NODE_BAND_KIND,
    Blake3Bn128WrapChunk = proofman_common::exec_format::BLAKE3_WRAP_CHUNK_BAND_KIND,
    Blake3Bn128WrapParent = proofman_common::exec_format::BLAKE3_WRAP_PARENT_BAND_KIND,
}

/// Result returned by every setup function, over the field `F` of the r1cs.
#[derive(Debug, Clone)]
pub struct SetupResult<F> {
    pub fixed_pols: Vec<FixedPol<F>>,
    pub pil_str: String,
    pub n_bits: usize,
    /// The size the circuit would take on its own, before any `min_n_bits` floor. The floor states
    /// the size the whole recursion runs at; whether a circuit is intrinsically too big for
    /// recursive1 is a question about the circuit, so it has to be asked of this.
    pub n_bits_natural: usize,
    /// Number of rows actually used (before power-of-2 padding). Mirrors JS `NUsed`.
    pub n_used: usize,
    pub s_map: Vec<Vec<u32>>,
    /// Row bands whose interior cells a trace expander recomputes from their boundary cells
    /// rather than gathering them out of the circom witness. One per hash-gate application --
    /// the only gates whose row band is wider than its inputs and outputs -- and, in the BN128
    /// wrap, one per range-check row, whose chunks its witness counts.
    pub gate_bands: Vec<GateBand>,
    pub plonk_additions: Vec<PlonkAddition<F>>,
    pub airgroup_name: String,
    pub air_name: String,
    /// A per-AIR parameter the band expander needs but cannot infer.
    ///
    /// BLAKE3 puts LANES here. It is a setup parameter, so deriving it from the column count would
    /// be reading a decision back out of its own consequence -- and that arithmetic means nothing
    /// for an air of another family. The poseidon setups write 0. The BN128 wrap writes the
    /// stage-1 column of its range checks' multiplicity, which its witness fills, if it has
    /// range-check rows.
    pub band_aux: u64,
}

// ─── Binary parser ───────────────────────────────────────────────────────────

fn read_u32(c: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    c.read_exact(&mut buf).map_err(|e| anyhow!("read u32: {}", e))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(c: &mut Cursor<&[u8]>) -> Result<u64> {
    let mut buf = [0u8; 8];
    c.read_exact(&mut buf).map_err(|e| anyhow!("read u64: {}", e))?;
    Ok(u64::from_le_bytes(buf))
}

/// The next `n` bytes, borrowed. Bounded by the data before anything is taken, so a corrupt length
/// is an error rather than an allocation of that size.
fn read_slice<'a>(c: &mut Cursor<&'a [u8]>, n: usize) -> Result<&'a [u8]> {
    let data: &'a [u8] = c.get_ref();
    let start = c.position() as usize;
    let end = start.checked_add(n).filter(|&end| end <= data.len());
    let end = end.ok_or_else(|| anyhow!("read {} bytes at {}: the data ends at {}", n, start, data.len()))?;
    c.set_position(end as u64);
    Ok(&data[start..end])
}

/// Read one element of `F`: `n8` bytes, little-endian, canonical.
fn read_field<F: PlonkField>(c: &mut Cursor<&[u8]>, n8: usize) -> Result<F> {
    let bytes = read_slice(c, n8)?;
    F::from_canonical_le(bytes).ok_or_else(|| {
        anyhow!("{} is not an element of {}: it is not below the prime", BigUint::from_bytes_le(bytes), F::PRIME)
    })
}

/// Read a null-terminated ASCII string.
fn read_cstring(c: &mut Cursor<&[u8]>) -> Result<String> {
    let mut bytes: Vec<u8> = Vec::new();
    loop {
        let mut b = [0u8; 1];
        c.read_exact(&mut b).map_err(|e| anyhow!("read cstring byte: {}", e))?;
        if b[0] == 0 {
            break;
        }
        bytes.push(b[0]);
    }
    String::from_utf8(bytes).map_err(|e| anyhow!("cstring is not valid UTF-8: {}", e))
}

/// Read a linear combination: n_terms × (wire_id u32, coeff n8 bytes).
fn read_lc<F: PlonkField>(c: &mut Cursor<&[u8]>, n8: usize) -> Result<LinearCombination<F>> {
    let n_terms = read_u32(c)? as usize;
    let mut lc = LinearCombination::new();
    for _ in 0..n_terms {
        let wire_id = read_u32(c)?;
        let coeff = read_field(c, n8)?;
        lc.insert(wire_id, coeff);
    }
    Ok(lc)
}

/// The byte position of each section's data, and whether the file has the custom-gate sections.
///
/// R1CS binary layout:
/// ```text
/// magic:      4 bytes  "r1cs"
/// version:    u32 LE   (= 1)
/// n_sections: u32 LE
/// for each section:
///     type:   u32 LE
///     size:   u64 LE   (byte count of data)
///     data:   size bytes
/// ```
fn read_sections(c: &mut Cursor<&[u8]>) -> Result<(HashMap<u32, u64>, bool)> {
    // Magic
    let magic = read_slice(c, 4)?;
    if magic != b"r1cs" {
        bail!("Invalid R1CS magic (expected \"r1cs\", got {:?})", magic);
    }

    // Version
    let version = read_u32(c)?;
    if version != 1 {
        bail!("Unsupported R1CS version: {} (expected 1)", version);
    }

    // Section count
    let n_sections = read_u32(c)? as usize;
    let use_custom_gates = n_sections == 5;

    // One-pass scan: record the byte position of each section's data.
    let mut data_starts: HashMap<u32, u64> = HashMap::new();
    for _ in 0..n_sections {
        let sec_type = read_u32(c)?;
        let sec_size = read_u64(c)?;
        data_starts.insert(sec_type, c.position());
        c.seek(SeekFrom::Current(sec_size as i64))
            .map_err(|e| anyhow!("section {}: seek past data: {}", sec_type, e))?;
    }
    Ok((data_starts, use_custom_gates))
}

/// Section 1, the header.
fn read_header(c: &mut Cursor<&[u8]>, data_starts: &HashMap<u32, u64>, use_custom_gates: bool) -> Result<R1csHeader> {
    c.set_position(*data_starts.get(&1).ok_or_else(|| anyhow!("missing header section (type 1)"))?);
    let n8 = read_u32(c)?;
    let prime_bytes = read_slice(c, n8 as usize)?.to_vec();
    Ok(R1csHeader {
        n8,
        prime_bytes,
        n_vars: read_u32(c)?,
        n_outputs: read_u32(c)?,
        n_pub_inputs: read_u32(c)?,
        n_prv_inputs: read_u32(c)?,
        n_labels: read_u64(c)?,
        n_constraints: read_u32(c)?,
        use_custom_gates,
    })
}

/// Parse only the header of an R1CS file: enough to tell which field it is over
/// ([`r1cs_prime`]) before reading it into one.
pub fn read_r1cs_header(data: &[u8]) -> Result<R1csHeader> {
    let mut c = Cursor::new(data);
    let (data_starts, use_custom_gates) = read_sections(&mut c)?;
    read_header(&mut c, &data_starts, use_custom_gates)
}

/// The prime an r1cs is over, or an error naming it if plonk2pil does not read that one.
pub fn r1cs_prime(header: &R1csHeader) -> Result<R1csPrime> {
    R1csPrime::from_modulus_le(&header.prime_bytes).ok_or_else(|| {
        anyhow!(
            "the r1cs is over the prime {} (n8 = {}), which plonk2pil does not read: it reads Goldilocks and BN128",
            BigUint::from_bytes_le(&header.prime_bytes),
            header.n8
        )
    })
}

/// Parse an R1CS file from raw bytes into `F`, the field it is over.
///
/// A file over another prime is refused, naming the prime: its elements would not be elements of
/// `F`, and on a narrower field, not even its coefficients' bytes.
pub fn read_r1cs_from_bytes<F: PlonkField>(data: &[u8]) -> Result<R1csFile<F>> {
    let mut c = Cursor::new(data);
    let (data_starts, use_custom_gates) = read_sections(&mut c)?;

    // ── Section 1: Header ───────────────────────────────────────────────────
    let header = read_header(&mut c, &data_starts, use_custom_gates)?;
    let prime = r1cs_prime(&header)?;
    ensure!(prime == F::PRIME, "the r1cs is over {prime}, and is read as {}", F::PRIME);
    // The prime's own width, since the prime is F's.
    let n8 = header.n8 as usize;
    let n_constraints = header.n_constraints;
    let n_vars = header.n_vars;

    // ── Section 2: Constraints ─────────────────────────────────────────────
    c.set_position(*data_starts.get(&2).ok_or_else(|| anyhow!("missing constraints section (type 2)"))?);
    let mut constraints = Vec::with_capacity(n_constraints as usize);
    for _ in 0..n_constraints {
        let a = read_lc(&mut c, n8)?;
        let b = read_lc(&mut c, n8)?;
        let lc_c = read_lc(&mut c, n8)?;
        constraints.push(R1csConstraint { a, b, c: lc_c });
    }

    // ── Section 3: Wire-to-label (optional) ───────────────────────────────
    let wire_to_label = if let Some(&pos) = data_starts.get(&3) {
        c.set_position(pos);
        let mut v = Vec::with_capacity(n_vars as usize);
        for _ in 0..n_vars {
            v.push(read_u64(&mut c)?);
        }
        v
    } else {
        Vec::new()
    };

    // ── Sections 4 & 5: Custom gates (only present when n_sections == 5) ──
    let mut custom_gates: Vec<CustomGate<F>> = Vec::new();
    let mut custom_gates_uses: Vec<CustomGateUse> = Vec::new();

    if use_custom_gates {
        // Section 4: gate definitions (name + parameters)
        c.set_position(*data_starts.get(&4).ok_or_else(|| anyhow!("missing custom gates used section (type 4)"))?);
        let n_gates = read_u32(&mut c)? as usize;
        custom_gates = Vec::with_capacity(n_gates);
        for _ in 0..n_gates {
            let template_name = read_cstring(&mut c)?;
            let n_params = read_u32(&mut c)? as usize;
            let mut parameters = Vec::with_capacity(n_params);
            for _ in 0..n_params {
                parameters.push(read_field(&mut c, n8)?);
            }
            custom_gates.push(CustomGate { template_name, parameters });
        }

        // Section 5: gate applications (gate_id + signal list)
        c.set_position(*data_starts.get(&5).ok_or_else(|| anyhow!("missing custom gates applied section (type 5)"))?);
        let n_apps = read_u32(&mut c)? as usize;
        custom_gates_uses = Vec::with_capacity(n_apps);
        for _ in 0..n_apps {
            let id = read_u32(&mut c)?;
            let n_signals = read_u32(&mut c)? as usize;
            let mut signals = Vec::with_capacity(n_signals);
            for _ in 0..n_signals {
                signals.push(read_u64(&mut c)?);
            }
            custom_gates_uses.push(CustomGateUse { id, signals });
        }
    }

    // Circom emits the same constraint system in a different order from run to run (the multiset is
    // identical; only the serialization order moves). The plonk placement is order-sensitive, so
    // canonicalise here and the proving key -- and its verkey -- become reproducible.
    constraints.sort_unstable();
    // The custom-gate uses need it too: the compressor/aggregation setups walk this vector
    // positionally and assign each use its own row band, so its order is the row assignment.
    custom_gates_uses.sort_unstable();

    Ok(R1csFile { header, constraints, wire_to_label, custom_gates, custom_gates_uses })
}
