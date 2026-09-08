//! The pil2-fflonk proving key: reader and writer.
//!
//! Layout is the `zkey` binfile container -- magic `"zkey"`, a `u32` version, a
//! `u32` section count, then `(id: u32, size: u64, data[size])` per section.
//! Twelve sections, split by role:
//!
//! * **Metadata** (1-5, 11): protocol id, field/curve parameters, the `f_i`
//!   packing, precomputed commitments, polynomial names and roots of unity.
//!   Parsed into typed values here.
//! * **Bulk** (6-10, 12): constant polynomial evaluations and coefficients,
//!   `x_n`, `x_2ns` and the powers of tau. Large opaque buffers, carried
//!   through byte-for-byte rather than interpreted.
//!
//! The container codec is local rather than reused from `pil2-stark-setup`'s
//! `BinFileWriter`: depending on that crate would create a cycle once it grows
//! a `setup_pilfflonk` command (see the crate docs), and the format is small
//! enough that duplicating it costs less than the coupling.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};

use crate::setup::{ShPlonkPol, ShPlonkStage, ShPlonkStagePol};

pub const MAGIC: &[u8; 4] = b"zkey";
pub const VERSION: u32 = 1;
pub const PROTOCOL_ID: u32 = 12;
pub const N_SECTIONS: u32 = 12;

pub const SECTION_ZKEY_HEADER: u32 = 1;
pub const SECTION_PF_HEADER: u32 = 2;
pub const SECTION_F: u32 = 3;
pub const SECTION_F_COMMITMENTS: u32 = 4;
pub const SECTION_POLS_NAMES_STAGE: u32 = 5;
pub const SECTION_CONST_POLS_EVALS: u32 = 6;
pub const SECTION_CONST_POLS_COEFS: u32 = 7;
pub const SECTION_CONST_POLS_EVALS_EXT: u32 = 8;
pub const SECTION_X_N: u32 = 9;
pub const SECTION_X_EXT: u32 = 10;
pub const SECTION_OMEGAS: u32 = 11;
pub const SECTION_PTAU: u32 = 12;

/// Sections carried through without interpretation.
pub const BULK_SECTIONS: [u32; 6] = [
    SECTION_CONST_POLS_EVALS,
    SECTION_CONST_POLS_COEFS,
    SECTION_CONST_POLS_EVALS_EXT,
    SECTION_X_N,
    SECTION_X_EXT,
    SECTION_PTAU,
];

/// Size of one field element as the zkey stores it: 32 bytes.
///
/// ffiasm has two BN254 Fr types and only one of them appears here. `FrElement`
/// in `fr.hpp` is the C API's tagged struct -- `{int32 shortVal; uint32 type;
/// uint64 longVal[4]}`, 40 bytes. `RawFr::Element` is `uint64_t[4]`, always in
/// Montgomery form, and that is what `AltBn128::Engine::FrElement` aliases, so
/// it is what every engine buffer and the zkey carry.
pub const FR_ELEMENT_BYTES: usize = 32;

/// A precomputed commitment to an `f_i` built only from constant polynomials.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FCommitment {
    pub name: String,
    /// Affine G1 point, two field elements, 64 bytes.
    pub commit: Vec<u8>,
    /// The committed polynomial's coefficients, as raw `FrElement`s.
    pub pol: Vec<u8>,
}

/// A parsed proving key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZKey {
    pub n8q: u32,
    pub q: Vec<u8>,
    pub n8r: u32,
    pub r: Vec<u8>,
    pub power: u32,
    pub power_w: u32,
    pub n_publics: u32,
    pub max_q_degree: u32,
    /// G2 element, `n8q * 4` bytes.
    pub x2: Vec<u8>,
    pub f: Vec<ShPlonkPol>,
    pub f_commitments: Vec<FCommitment>,
    /// Stage to the polynomial names it contributes, in file order. The writer
    /// drops the per-name indices and recovers them by position.
    pub pols_names_stage: BTreeMap<u32, Vec<String>>,
    /// Roots of unity, in file order, each an untouched `FrElement`.
    pub omegas: Vec<(String, Vec<u8>)>,
    /// Bulk sections, by id.
    pub bulk: BTreeMap<u32, Vec<u8>>,
}

// ---------------------------------------------------------------- reading

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            bail!("truncated: want {n} bytes at offset {}, have {}", self.pos, self.buf.len() - self.pos);
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// Null-terminated, as `BinFileWriter::write_string` emits.
    fn string(&mut self) -> Result<String> {
        let start = self.pos;
        while self.pos < self.buf.len() && self.buf[self.pos] != 0 {
            self.pos += 1;
        }
        if self.pos >= self.buf.len() {
            bail!("unterminated string at offset {start}");
        }
        let s = std::str::from_utf8(&self.buf[start..self.pos]).context("string is not utf-8")?.to_string();
        self.pos += 1; // the NUL
        Ok(s)
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }
}

impl ZKey {
    /// Split a zkey container into its raw sections.
    pub fn split_sections(bytes: &[u8]) -> Result<BTreeMap<u32, Vec<u8>>> {
        let mut c = Cursor::new(bytes);

        let magic = c.take(4)?;
        if magic != MAGIC {
            bail!("bad magic {magic:?}, want {MAGIC:?}");
        }
        let version = c.u32()?;
        if version != VERSION {
            bail!("unsupported zkey version {version}, want {VERSION}");
        }
        let n_sections = c.u32()?;

        let mut sections = BTreeMap::new();
        for _ in 0..n_sections {
            let id = c.u32()?;
            let size = c.u64()? as usize;
            let data = c.take(size)?.to_vec();
            if sections.insert(id, data).is_some() {
                bail!("duplicate section {id}");
            }
        }
        Ok(sections)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut sections = Self::split_sections(bytes)?;

        // Section 1: the protocol id, so a foreign zkey fails loudly here
        // rather than as a misparse further in.
        let header = sections.remove(&SECTION_ZKEY_HEADER).context("missing zkey header section")?;
        let protocol = Cursor::new(&header).u32().context("reading protocol id")?;
        if protocol != PROTOCOL_ID {
            bail!("zkey declares protocol {protocol}, want {PROTOCOL_ID} (pil-fflonk)");
        }

        let pf = sections.remove(&SECTION_PF_HEADER).context("missing pil-fflonk header section")?;
        let mut c = Cursor::new(&pf);
        let n8q = c.u32()?;
        let q = c.take(n8q as usize)?.to_vec();
        let n8r = c.u32()?;
        let r = c.take(n8r as usize)?.to_vec();
        let power = c.u32()?;
        let power_w = c.u32()?;
        let n_publics = c.u32()?;
        let max_q_degree = c.u32()?;
        let x2 = c.take(n8q as usize * 4)?.to_vec();
        if !c.done() {
            bail!("pil-fflonk header section has {} trailing bytes", pf.len() - c.pos);
        }

        let f = read_f(&sections.remove(&SECTION_F).context("missing f section")?)?;
        let f_commitments =
            read_f_commitments(&sections.remove(&SECTION_F_COMMITMENTS).context("missing f commitments section")?)?;
        let pols_names_stage = read_pols_names_stage(
            &sections.remove(&SECTION_POLS_NAMES_STAGE).context("missing pols names stage section")?,
        )?;
        let omegas = read_omegas(&sections.remove(&SECTION_OMEGAS).context("missing omegas section")?)?;

        // Everything left must be a known bulk section.
        for id in sections.keys() {
            if !BULK_SECTIONS.contains(id) {
                bail!("unrecognised zkey section {id}");
            }
        }

        Ok(Self {
            n8q,
            q,
            n8r,
            r,
            power,
            power_w,
            n_publics,
            max_q_degree,
            x2,
            f,
            f_commitments,
            pols_names_stage,
            omegas,
            bulk: sections,
        })
    }
}

fn read_f(buf: &[u8]) -> Result<Vec<ShPlonkPol>> {
    let mut c = Cursor::new(buf);
    let len = c.u32()?;
    let mut out = Vec::with_capacity(len as usize);

    for _ in 0..len {
        let index = c.u32()?;
        let degree = c.u32()? as u64;

        let n_opening_points = c.u32()?;
        let mut opening_points = Vec::with_capacity(n_opening_points as usize);
        for _ in 0..n_opening_points {
            opening_points.push(c.u32()?);
        }

        let n_pols = c.u32()?;
        let mut pols = Vec::with_capacity(n_pols as usize);
        for _ in 0..n_pols {
            pols.push(c.string()?);
        }

        let n_stages = c.u32()?;
        let mut stages = Vec::with_capacity(n_stages as usize);
        for _ in 0..n_stages {
            let stage = c.u32()?;
            let stage_n_pols = c.u32()?;
            let mut stage_pols = Vec::with_capacity(stage_n_pols as usize);
            for _ in 0..stage_n_pols {
                let name = c.string()?;
                let degree = c.u32()? as u64;
                stage_pols.push(ShPlonkStagePol { name, degree });
            }
            stages.push(ShPlonkStage { stage, pols: stage_pols });
        }

        out.push(ShPlonkPol { index, degree, opening_points, pols, stages });
    }

    if !c.done() {
        bail!("f section has {} trailing bytes", buf.len() - c.pos);
    }
    Ok(out)
}

fn read_f_commitments(buf: &[u8]) -> Result<Vec<FCommitment>> {
    let mut c = Cursor::new(buf);
    let len = c.u32()?;
    let mut out = Vec::with_capacity(len as usize);

    for _ in 0..len {
        let name = c.string()?;
        let commit = c.take(64)?.to_vec();
        let pol_bytes = c.u32()? as usize;
        let pol = c.take(pol_bytes)?.to_vec();
        out.push(FCommitment { name, commit, pol });
    }

    if !c.done() {
        bail!("f commitments section has {} trailing bytes", buf.len() - c.pos);
    }
    Ok(out)
}

fn read_pols_names_stage(buf: &[u8]) -> Result<BTreeMap<u32, Vec<String>>> {
    let mut c = Cursor::new(buf);
    let len = c.u32()?;
    let mut out = BTreeMap::new();

    for _ in 0..len {
        let stage = c.u32()?;
        let n_names = c.u32()?;
        let mut names = Vec::with_capacity(n_names as usize);
        for _ in 0..n_names {
            names.push(c.string()?);
        }
        if out.insert(stage, names).is_some() {
            bail!("duplicate stage {stage} in pols names stage section");
        }
    }

    if !c.done() {
        bail!("pols names stage section has {} trailing bytes", buf.len() - c.pos);
    }
    Ok(out)
}

fn read_omegas(buf: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
    let mut c = Cursor::new(buf);
    let len = c.u32()?;
    let mut out = Vec::with_capacity(len as usize);

    for _ in 0..len {
        let name = c.string()?;
        let value = c.take(FR_ELEMENT_BYTES)?.to_vec();
        out.push((name, value));
    }

    if !c.done() {
        bail!("omegas section has {} trailing bytes", buf.len() - c.pos);
    }
    Ok(out)
}

// ---------------------------------------------------------------- writing

/// The container writer: sections are buffered, then emitted with their sizes.
#[derive(Default)]
struct BinFileBuilder {
    sections: Vec<(u32, Vec<u8>)>,
}

impl BinFileBuilder {
    fn section(&mut self, id: u32, data: Vec<u8>) {
        self.sections.push((id, data));
    }

    /// A complete key carries all [`N_SECTIONS`]. A short write means a bulk
    /// section went missing between read and write, which would produce a file
    /// the C++ reader rejects only once it looks for that section.
    fn finish_complete(self) -> Vec<u8> {
        assert_eq!(
            self.sections.len(),
            N_SECTIONS as usize,
            "zkey must have {N_SECTIONS} sections, built {}",
            self.sections.len()
        );
        self.finish()
    }

    fn finish(self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.sections.len() as u32).to_le_bytes());
        for (id, data) in self.sections {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&(data.len() as u64).to_le_bytes());
            out.extend_from_slice(&data);
        }
        out
    }
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(s.as_bytes());
    buf.push(0);
}

impl ZKey {
    /// Serialise back to the zkey container.
    ///
    /// Section order follows the ids, which is the order pil-fflonk's writer
    /// emits them in.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = BinFileBuilder::default();

        b.section(SECTION_ZKEY_HEADER, PROTOCOL_ID.to_le_bytes().to_vec());

        let mut pf = Vec::new();
        put_u32(&mut pf, self.n8q);
        pf.extend_from_slice(&self.q);
        put_u32(&mut pf, self.n8r);
        pf.extend_from_slice(&self.r);
        put_u32(&mut pf, self.power);
        put_u32(&mut pf, self.power_w);
        put_u32(&mut pf, self.n_publics);
        put_u32(&mut pf, self.max_q_degree);
        pf.extend_from_slice(&self.x2);
        b.section(SECTION_PF_HEADER, pf);

        let mut f = Vec::new();
        put_u32(&mut f, self.f.len() as u32);
        for fi in &self.f {
            put_u32(&mut f, fi.index);
            put_u32(&mut f, fi.degree as u32);
            put_u32(&mut f, fi.opening_points.len() as u32);
            for p in &fi.opening_points {
                put_u32(&mut f, *p);
            }
            put_u32(&mut f, fi.pols.len() as u32);
            for name in &fi.pols {
                put_string(&mut f, name);
            }
            put_u32(&mut f, fi.stages.len() as u32);
            for stage in &fi.stages {
                put_u32(&mut f, stage.stage);
                put_u32(&mut f, stage.pols.len() as u32);
                for pol in &stage.pols {
                    put_string(&mut f, &pol.name);
                    put_u32(&mut f, pol.degree as u32);
                }
            }
        }
        b.section(SECTION_F, f);

        let mut fc = Vec::new();
        put_u32(&mut fc, self.f_commitments.len() as u32);
        for c in &self.f_commitments {
            put_string(&mut fc, &c.name);
            fc.extend_from_slice(&c.commit);
            put_u32(&mut fc, c.pol.len() as u32);
            fc.extend_from_slice(&c.pol);
        }
        b.section(SECTION_F_COMMITMENTS, fc);

        let mut pns = Vec::new();
        put_u32(&mut pns, self.pols_names_stage.len() as u32);
        for (stage, names) in &self.pols_names_stage {
            put_u32(&mut pns, *stage);
            put_u32(&mut pns, names.len() as u32);
            for name in names {
                put_string(&mut pns, name);
            }
        }
        b.section(SECTION_POLS_NAMES_STAGE, pns);

        for id in [
            SECTION_CONST_POLS_EVALS,
            SECTION_CONST_POLS_COEFS,
            SECTION_CONST_POLS_EVALS_EXT,
            SECTION_X_N,
            SECTION_X_EXT,
        ] {
            if let Some(data) = self.bulk.get(&id) {
                b.section(id, data.clone());
            }
        }

        let mut om = Vec::new();
        put_u32(&mut om, self.omegas.len() as u32);
        for (name, value) in &self.omegas {
            put_string(&mut om, name);
            om.extend_from_slice(value);
        }
        b.section(SECTION_OMEGAS, om);

        if let Some(data) = self.bulk.get(&SECTION_PTAU) {
            b.section(SECTION_PTAU, data.clone());
        }

        b.finish_complete()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::ShPlonkSetup;

    /// A minimal but structurally complete key, so the round-trip exercises
    /// every metadata section without needing the 1 MB fixture.
    fn synthetic() -> ZKey {
        ZKey {
            n8q: 32,
            q: vec![7u8; 32],
            n8r: 32,
            r: vec![9u8; 32],
            power: 8,
            power_w: 12,
            n_publics: 3,
            max_q_degree: 0,
            x2: vec![1u8; 128],
            f: vec![
                ShPlonkPol {
                    index: 0,
                    degree: 1541,
                    opening_points: vec![0],
                    pols: vec!["A.x".into(), "A.y".into()],
                    stages: vec![ShPlonkStage {
                        stage: 0,
                        pols: vec![
                            ShPlonkStagePol { name: "A.x".into(), degree: 256 },
                            ShPlonkStagePol { name: "A.y".into(), degree: 256 },
                        ],
                    }],
                },
                ShPlonkPol {
                    index: 1,
                    degree: 780,
                    opening_points: vec![0, 1],
                    pols: vec!["Q".into()],
                    stages: vec![ShPlonkStage {
                        stage: 4,
                        pols: vec![ShPlonkStagePol { name: "Q".into(), degree: 780 }],
                    }],
                },
            ],
            f_commitments: vec![FCommitment {
                name: "f0".into(),
                commit: vec![3u8; 64],
                pol: vec![5u8; FR_ELEMENT_BYTES * 2],
            }],
            pols_names_stage: BTreeMap::from([
                (0u32, vec!["A.x".to_string(), "A.y".to_string()]),
                (4u32, vec!["Q".to_string()]),
            ]),
            omegas: vec![
                ("w".to_string(), vec![11u8; FR_ELEMENT_BYTES]),
                ("w1".to_string(), vec![12u8; FR_ELEMENT_BYTES]),
                ("w2_1d2".to_string(), vec![13u8; FR_ELEMENT_BYTES]),
            ],
            bulk: BTreeMap::from([
                (SECTION_CONST_POLS_EVALS, vec![21u8; 96]),
                (SECTION_CONST_POLS_COEFS, vec![22u8; 96]),
                (SECTION_CONST_POLS_EVALS_EXT, vec![23u8; 192]),
                (SECTION_X_N, vec![24u8; 64]),
                (SECTION_X_EXT, vec![25u8; 128]),
                (SECTION_PTAU, vec![26u8; 256]),
            ]),
        }
    }

    #[test]
    fn round_trips_through_bytes() {
        let key = synthetic();
        let parsed = ZKey::from_bytes(&key.to_bytes()).unwrap();
        assert_eq!(parsed, key);
    }

    #[test]
    fn written_bytes_are_stable() {
        // Writing what we read must reproduce the same bytes, or a
        // read-modify-write cycle would perturb keys it did not change.
        let bytes = synthetic().to_bytes();
        let again = ZKey::from_bytes(&bytes).unwrap().to_bytes();
        assert_eq!(again, bytes);
    }

    #[test]
    fn container_header_matches_the_format() {
        let bytes = synthetic().to_bytes();
        assert_eq!(&bytes[0..4], MAGIC);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), VERSION);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), N_SECTIONS);
    }

    #[test]
    fn pf_header_section_is_216_bytes() {
        // n8q(4) + q(32) + n8r(4) + r(32) + 4 x u32(16) + X2(128) = 216, the
        // size the real pil-fflonk zkey carries.
        let sections = ZKey::split_sections(&synthetic().to_bytes()).unwrap();
        assert_eq!(sections[&SECTION_PF_HEADER].len(), 216);
    }

    #[test]
    fn rejects_a_foreign_protocol_id() {
        let mut bytes = synthetic().to_bytes();
        // Section 1's payload starts 12 (container) + 12 (section header) in.
        bytes[24..28].copy_from_slice(&10u32.to_le_bytes());
        let err = ZKey::from_bytes(&bytes).unwrap_err().to_string();
        assert!(err.contains("protocol 10"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = synthetic().to_bytes();
        bytes[0..4].copy_from_slice(b"vkey");
        assert!(ZKey::from_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_truncation() {
        let bytes = synthetic().to_bytes();
        assert!(ZKey::from_bytes(&bytes[..bytes.len() - 8]).is_err());
    }

    /// The zkey and vkey are produced separately by pil-fflonk's setup, so
    /// agreeing on the packing is a real cross-check rather than a tautology.
    /// Reads the zkey from the pil-fflonk checkout when present -- it is ~1 MB,
    /// too large to vendor, unlike the 6 KB vkey.
    #[test]
    fn real_zkey_agrees_with_the_vkey_fixture() {
        let path = std::path::Path::new("/home/xavi/dev/pil-fflonk/config/pilfflonk.zkey");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }

        let bytes = std::fs::read(path).unwrap();
        let zkey = ZKey::from_bytes(&bytes).expect("real zkey parses");

        let vkey: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/pilfflonk.vkey")).unwrap();
        let setup = ShPlonkSetup::from_vkey_json(&vkey).unwrap();

        assert_eq!(zkey.power, setup.power);
        assert_eq!(zkey.power_w, setup.power_w);
        assert_eq!(zkey.n_publics, setup.n_publics);
        assert_eq!(zkey.max_q_degree, setup.max_q_degree);
        assert_eq!(zkey.f.len(), setup.f.len());
        assert_eq!(zkey.f, setup.f, "the f_i packing must match between zkey and vkey");
        assert_eq!(zkey.f_commitments.len(), setup.f_commitments.len());
        // The vkey adds a top-level "w" that the zkey's omegas map does not
        // carry, so it holds exactly one key more.
        assert_eq!(setup.omegas.len(), zkey.omegas.len() + 1);
        assert!(setup.omegas.contains_key("w"));
        for (name, _) in &zkey.omegas {
            assert!(setup.omegas.contains_key(name), "vkey is missing omega {name:?}");
        }

        // And the parse must be exact enough to rebuild the file.
        assert_eq!(zkey.to_bytes(), bytes, "re-emitting the real zkey must be byte-identical");
    }
}
