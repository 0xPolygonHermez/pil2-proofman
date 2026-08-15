//! R1CS types and direct `&[u8]` parser for use by plonk2pil setup routines.
//!
//! All field elements are u64 (Goldilocks fits in 8 bytes).
//! No BigInt, no temporary files.

use std::collections::HashMap;
use std::io::{Cursor, Read, Seek, SeekFrom};

use anyhow::{anyhow, bail, Result};

// ─── Public types ───────────────────────────────────────────────────────────

/// Wire index → Goldilocks coefficient.
pub type LinearCombination = HashMap<u32, u64>;

/// A single R1CS constraint: A * B = C.
#[derive(Debug, Clone)]
pub struct R1csConstraint {
    pub a: LinearCombination,
    pub b: LinearCombination,
    pub c: LinearCombination,
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
pub struct CustomGate {
    pub template_name: String,
    pub parameters: Vec<u64>,
}

/// One application of a custom gate (gate index + wire signals).
#[derive(Debug, Clone)]
pub struct CustomGateUse {
    pub id: u32,
    pub signals: Vec<u64>,
}

/// Complete parsed R1CS file.
#[derive(Debug, Clone)]
pub struct R1csFile {
    pub header: R1csHeader,
    pub constraints: Vec<R1csConstraint>,
    pub wire_to_label: Vec<u64>,
    pub custom_gates: Vec<CustomGate>,
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
}

impl Default for PlonkOptions {
    fn default() -> Self {
        Self {
            airgroup_name: None,
            max_constraint_degree: None,
            hash_id: proofman_common::hash_family::DEFAULT_HASH_ID.to_string(),
            merge_copies: true,
        }
    }
}

/// A single fixed polynomial column.
#[derive(Debug, Clone)]
pub struct FixedPol {
    pub name: String,
    pub index: usize,
    pub values: Vec<u64>,
}

/// Result returned by every setup function.
#[derive(Debug, Clone)]
pub struct SetupResult {
    pub fixed_pols: Vec<FixedPol>,
    pub pil_str: String,
    pub n_bits: usize,
    /// Number of rows actually used (before power-of-2 padding). Mirrors JS `NUsed`.
    pub n_used: usize,
    pub s_map: Vec<Vec<u32>>,
    pub plonk_additions: Vec<[u64; 4]>,
    pub airgroup_name: String,
    pub air_name: String,
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

/// Read `field_size` bytes (≤ 8) as a little-endian u64.
fn read_field(c: &mut Cursor<&[u8]>, field_size: usize) -> Result<u64> {
    debug_assert!(field_size <= 8);
    let mut buf = [0u8; 8];
    c.read_exact(&mut buf[..field_size]).map_err(|e| anyhow!("read field ({} bytes): {}", field_size, e))?;
    Ok(u64::from_le_bytes(buf))
}

fn read_bytes_vec(c: &mut Cursor<&[u8]>, n: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    c.read_exact(&mut buf).map_err(|e| anyhow!("read {} bytes: {}", n, e))?;
    Ok(buf)
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

/// Read a linear combination: n_terms × (wire_id u32, coeff field_size bytes).
fn read_lc(c: &mut Cursor<&[u8]>, field_size: usize) -> Result<LinearCombination> {
    let n_terms = read_u32(c)? as usize;
    let mut lc = HashMap::with_capacity(n_terms);
    for _ in 0..n_terms {
        let wire_id = read_u32(c)?;
        let coeff = read_field(c, field_size)?;
        lc.insert(wire_id, coeff);
    }
    Ok(lc)
}

/// Parse an R1CS file from raw bytes.
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
pub fn read_r1cs_from_bytes(data: &[u8]) -> Result<R1csFile> {
    let mut c = Cursor::new(data);

    // Magic
    let magic = read_bytes_vec(&mut c, 4)?;
    if magic != b"r1cs" {
        bail!("Invalid R1CS magic (expected \"r1cs\", got {:?})", magic);
    }

    // Version
    let version = read_u32(&mut c)?;
    if version != 1 {
        bail!("Unsupported R1CS version: {} (expected 1)", version);
    }

    // Section count
    let n_sections = read_u32(&mut c)? as usize;
    let use_custom_gates = n_sections == 5;

    // One-pass scan: record the byte position of each section's data.
    let mut data_starts: HashMap<u32, u64> = HashMap::new();
    for _ in 0..n_sections {
        let sec_type = read_u32(&mut c)?;
        let sec_size = read_u64(&mut c)?;
        data_starts.insert(sec_type, c.position());
        c.seek(SeekFrom::Current(sec_size as i64))
            .map_err(|e| anyhow!("section {}: seek past data: {}", sec_type, e))?;
    }

    // ── Section 1: Header ───────────────────────────────────────────────────
    c.set_position(*data_starts.get(&1).ok_or_else(|| anyhow!("missing header section (type 1)"))?);
    let field_size = read_u32(&mut c)? as usize;
    let prime_bytes = read_bytes_vec(&mut c, field_size)?;
    let n_vars = read_u32(&mut c)?;
    let n_outputs = read_u32(&mut c)?;
    let n_pub_inputs = read_u32(&mut c)?;
    let n_prv_inputs = read_u32(&mut c)?;
    let n_labels = read_u64(&mut c)?;
    let n_constraints = read_u32(&mut c)?;

    let header = R1csHeader {
        n8: field_size as u32,
        prime_bytes,
        n_vars,
        n_outputs,
        n_pub_inputs,
        n_prv_inputs,
        n_labels,
        n_constraints,
        use_custom_gates,
    };

    // ── Section 2: Constraints ─────────────────────────────────────────────
    c.set_position(*data_starts.get(&2).ok_or_else(|| anyhow!("missing constraints section (type 2)"))?);
    let mut constraints = Vec::with_capacity(n_constraints as usize);
    for _ in 0..n_constraints {
        let a = read_lc(&mut c, field_size)?;
        let b = read_lc(&mut c, field_size)?;
        let lc_c = read_lc(&mut c, field_size)?;
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
    let mut custom_gates: Vec<CustomGate> = Vec::new();
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
                parameters.push(read_field(&mut c, field_size)?);
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

    Ok(R1csFile { header, constraints, wire_to_label, custom_gates, custom_gates_uses })
}
