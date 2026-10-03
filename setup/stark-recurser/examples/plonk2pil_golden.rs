//! Driver of the plonk2pil golden, `setup/golden/plonk2pil` (see its README.md). It runs
//! [`plonk2pil`] on one r1cs, with every option given on the command line, and writes what it returns
//! in a byte form that changes only when an output does.
//!
//! ```text
//! plonk2pil_golden canonical-r1cs <IN> <OUT>
//! plonk2pil_golden run <R1CS> <OUT_DIR> --setup-type <compressor|aggregation> --hash <FAMILY>
//!     --airgroup <NAME> [--max-degree <N>] [--min-n-bits <N>] [--blake3-lanes <N>]
//! ```
//!
//! `canonical-r1cs` rewrites an r1cs with its sections in type order, the terms of each linear
//! combination by wire, and the constraints and the custom-gate uses sorted by their bytes. circom
//! writes the same system in a different order from run to run, so its r1cs cannot be hashed as it
//! is; the canonical one can, and plonk2pil reads both to the same `R1csFile`, since it keys each
//! combination by wire and sorts the constraints and the uses itself. The file is parsed here rather
//! than with plonk2pil's reader, so that the digest of an input does not move with the code under
//! test.
//!
//! `run` writes into `OUT_DIR`, which must not exist yet:
//! - `air.pil`: `pil_str`, as it is;
//! - `air.exec`: `exec` as u64 little-endian, the bytes the setups write to `<air>.exec`;
//! - `fixed/<name>.<index>.bin`: the values of one `fixed_pols` entry, as u64 little-endian;
//! - `result.json`: the other fields, and the `fixed_pols` entries in their order.
//!
//! `merge_copies` is always on, as every caller of plonk2pil sets it.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::str::FromStr;

use anyhow::{bail, ensure, Context, Result};
use pil2_stark_recurser::plonk2pil::r1cs_types::PlonkOptions;
use pil2_stark_recurser::plonk2pil::{plonk2pil, PlonkResult};
use serde_json::json;

const USAGE: &str = "usage:
  plonk2pil_golden canonical-r1cs <IN> <OUT>
  plonk2pil_golden run <R1CS> <OUT_DIR> --setup-type <compressor|aggregation> --hash <FAMILY>
      --airgroup <NAME> [--max-degree <N>] [--min-n-bits <N>] [--blake3-lanes <N>]";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((cmd, rest)) if cmd == "canonical-r1cs" => {
            let [input, output] = rest else { bail!("{USAGE}") };
            let data = fs::read(input).with_context(|| format!("read {input}"))?;
            let canonical = canonical_r1cs(&data).with_context(|| format!("canonicalize {input}"))?;
            fs::write(output, canonical).with_context(|| format!("write {output}"))
        }
        Some((cmd, rest)) if cmd == "run" => run(rest),
        _ => bail!("{USAGE}"),
    }
}

// ─── run ────────────────────────────────────────────────────────────────────

fn run(args: &[String]) -> Result<()> {
    let [r1cs, out, flags @ ..] = args else { bail!("{USAGE}") };

    let mut setup_type: Option<String> = None;
    let mut hash: Option<String> = None;
    let mut airgroup: Option<String> = None;
    let mut max_degree: Option<usize> = None;
    let mut min_n_bits: Option<usize> = None;
    let mut blake3_lanes: Option<usize> = None;
    let mut flags = flags.iter();
    while let Some(flag) = flags.next() {
        let value = flags.next().with_context(|| format!("{flag} needs a value\n{USAGE}"))?;
        match flag.as_str() {
            "--setup-type" => set_once(&mut setup_type, flag, value)?,
            "--hash" => set_once(&mut hash, flag, value)?,
            "--airgroup" => set_once(&mut airgroup, flag, value)?,
            "--max-degree" => set_once(&mut max_degree, flag, value)?,
            "--min-n-bits" => set_once(&mut min_n_bits, flag, value)?,
            "--blake3-lanes" => set_once(&mut blake3_lanes, flag, value)?,
            _ => bail!("unknown flag {flag}\n{USAGE}"),
        }
    }
    let setup_type = setup_type.context("--setup-type is required")?;
    // Required rather than defaulted: a default hash family or airgroup name would let a config
    // leave out the very options its outputs depend on.
    let options = PlonkOptions {
        airgroup_name: Some(airgroup.context("--airgroup is required")?),
        max_constraint_degree: max_degree,
        hash_id: hash.context("--hash is required")?,
        merge_copies: true,
        blake3_lanes,
        min_n_bits,
    };

    let data = fs::read(r1cs).with_context(|| format!("read {r1cs}"))?;
    let result = plonk2pil(&data, &setup_type, &options).with_context(|| format!("plonk2pil on {r1cs}"))?;
    write_outputs(&result, Path::new(out))
}

fn set_once<T: FromStr>(slot: &mut Option<T>, flag: &str, value: &str) -> Result<()>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    ensure!(slot.is_none(), "{flag} is given twice");
    *slot = Some(value.parse().with_context(|| format!("{flag} {value}"))?);
    Ok(())
}

fn write_outputs(result: &PlonkResult, out: &Path) -> Result<()> {
    ensure!(!out.exists(), "{} already exists", out.display());
    let fixed_dir = out.join("fixed");
    fs::create_dir_all(&fixed_dir).with_context(|| format!("create {}", fixed_dir.display()))?;

    write_new(&out.join("air.pil"), result.pil_str.as_bytes())?;
    write_new(&out.join("air.exec"), &words_le(&result.exec))?;

    let mut seen = BTreeSet::new();
    let mut fixed_pols = Vec::with_capacity(result.fixed_pols.len());
    for pol in &result.fixed_pols {
        ensure!(seen.insert((pol.name.as_str(), pol.index)), "fixed pol {}[{}] appears twice", pol.name, pol.index);
        ensure!(!pol.name.contains(['/', '\0']), "fixed pol name {:?} cannot be a file name", pol.name);
        let file = format!("{}.{}.bin", pol.name, pol.index);
        write_new(&fixed_dir.join(&file), &words_le(&pol.values))?;
        fixed_pols.push(json!({ "name": pol.name, "index": pol.index, "file": format!("fixed/{file}") }));
    }

    let summary = json!({
        "airgroupName": result.airgroup_name,
        "airName": result.air_name,
        "nBits": result.n_bits,
        "nBitsNatural": result.n_bits_natural,
        "nUsed": result.n_used,
        "fixedPols": fixed_pols,
    });
    write_new(&out.join("result.json"), format!("{}\n", serde_json::to_string_pretty(&summary)?).as_bytes())
}

/// Write a file that must not exist yet, so that two outputs mapped to one name fail loudly.
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::File::create_new(path).and_then(|mut f| f.write_all(bytes)).with_context(|| format!("write {}", path.display()))
}

fn words_le(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

// ─── canonical-r1cs ─────────────────────────────────────────────────────────

/// Section types: those of the iden3 r1cs format, and circom's two for custom gates.
const SECTION_HEADER: u32 = 1;
const SECTION_CONSTRAINTS: u32 = 2;
const SECTION_CUSTOM_GATES_USES: u32 = 5;

/// The r1cs `data` with its sections in type order, and the order-free contents of two of them
/// sorted: the constraints (and the terms of each combination) and the custom-gate uses. The other
/// sections are positional -- the header, the wire-to-label map, the custom-gate list whose index a
/// use names -- and are copied as they are.
fn canonical_r1cs(data: &[u8]) -> Result<Vec<u8>> {
    let mut r = Reader::new(data);
    ensure!(r.bytes(4)? == b"r1cs", "not an r1cs file");
    let version = r.u32()?;
    ensure!(version == 1, "r1cs version {version}, expected 1");
    let n_sections = r.u32()?;
    let mut sections = BTreeMap::new();
    for _ in 0..n_sections {
        let kind = r.u32()?;
        let size = usize::try_from(r.u64()?)?;
        ensure!(sections.insert(kind, r.bytes(size)?).is_none(), "section {kind} appears twice");
    }
    ensure!(r.is_empty(), "bytes after the last section");

    let mut header = Reader::new(sections.get(&SECTION_HEADER).context("no header section")?);
    let n8 = header.u32()? as usize;
    header.bytes(n8)?; // the prime
    header.bytes(4 * 4 + 8)?; // nWires, nPubOut, nPubIn, nPrvIn, nLabels
    let n_constraints = header.u32()?;
    ensure!(header.is_empty(), "header section: unexpected trailing bytes");

    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(b"r1cs");
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&n_sections.to_le_bytes());
    for (&kind, &body) in &sections {
        let body = match kind {
            SECTION_CONSTRAINTS => canonical_constraints(body, n8, n_constraints)?,
            SECTION_CUSTOM_GATES_USES => canonical_gate_uses(body)?,
            _ => body.to_vec(),
        };
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(&(body.len() as u64).to_le_bytes());
        out.extend_from_slice(&body);
    }
    Ok(out)
}

/// `n` constraints `A * B = C`, each combination `nTerms: u32` then `(wire: u32, coef: n8 bytes)`
/// per term: the terms of each combination sorted by wire, then the constraints by their bytes.
fn canonical_constraints(body: &[u8], n8: usize, n: u32) -> Result<Vec<u8>> {
    let mut r = Reader::new(body);
    let mut constraints = Vec::with_capacity(n as usize);
    for i in 0..n {
        let mut constraint = Vec::new();
        for _ in 0..3 {
            let n_terms = r.u32()?;
            let mut terms = Vec::with_capacity(n_terms as usize);
            for _ in 0..n_terms {
                terms.push((r.u32()?, r.bytes(n8)?));
            }
            terms.sort_unstable_by_key(|&(wire, _)| wire);
            // plonk2pil keeps the last of two terms on one wire, so their order would matter.
            if let Some(w) = terms.windows(2).find(|w| w[0].0 == w[1].0) {
                bail!("constraint {i}: a combination has two terms on wire {}", w[0].0);
            }
            constraint.extend_from_slice(&n_terms.to_le_bytes());
            for (wire, coef) in terms {
                constraint.extend_from_slice(&wire.to_le_bytes());
                constraint.extend_from_slice(coef);
            }
        }
        constraints.push(constraint);
    }
    ensure!(r.is_empty(), "constraints section: bytes after constraint {n}");
    constraints.sort_unstable();
    Ok(constraints.concat())
}

/// `nUses: u32`, then per use `gate: u32, nSignals: u32` and the signals as u64 each, sorted by
/// their bytes. A use's signals are the gate's inputs and outputs in order, so they stay as they are.
fn canonical_gate_uses(body: &[u8]) -> Result<Vec<u8>> {
    let mut r = Reader::new(body);
    let n = r.u32()?;
    let mut uses = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let start = r.pos;
        r.u32()?; // the gate
        let n_signals = r.u32()? as usize;
        r.bytes(n_signals * 8)?;
        uses.push(&body[start..r.pos]);
    }
    ensure!(r.is_empty(), "custom gate uses section: bytes after use {n}");
    uses.sort_unstable();
    let mut out = n.to_le_bytes().to_vec();
    out.extend(uses.concat());
    Ok(out)
}

/// Little-endian reads off a byte slice, failing at its end.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos == self.data.len()
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&end| end <= self.data.len());
        let end = end.with_context(|| format!("{n} bytes at offset {}: past the end", self.pos))?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into()?))
    }
}
