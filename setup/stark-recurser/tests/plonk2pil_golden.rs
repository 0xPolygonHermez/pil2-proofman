//! Golden harness for plonk2pil and its checker over every r1cs case of `GOLDEN_DIR`. Run as
//! `GOLDEN_DIR=<cases> GOLDEN_OUT=<out> [GOLDEN_BASELINE=<out of an earlier run>] [GOLDEN_SIZES_ONLY=1]
//!  [PCHECK_BUILDS=<dir of the b-<hash>, rt-<hash>-<kind> setups>] [PCHECK_ONLY=<substr>]
//!  cargo test --release -p pil2-stark-recurser --test plonk2pil_golden -- --ignored --nocapture`.
//!
//! A case is `GOLDEN_DIR/<name>/{circuit.r1cs, meta.json}`; `meta.json` is
//! `{"family": "blake3", "kind": "aggregation", "lanes": null, "min_n_bits": null, "max_degree": null,
//!   "airgroup": null}` (`kind`: compressor | aggregation | wrap; a wrap's family is its hash).
//! `GOLDEN_OUT` gets `sizes.tsv` and `hashes.tsv` (sha256 of `exec`, `pil_str` and every fixed
//! column). With a baseline it prints `OK|FAIL <case> <first differing artifact>`; byte mode compares
//! sizes and hashes, `GOLDEN_SIZES_ONLY` is the size gate (nBits equal; NUsed, S and plonk
//! constraints not greater). With `PCHECK_BUILDS` it also runs the checker on each result against
//! the pilout its setup compiled: `CHECK OK|FAIL|STALE <case> <report|error>`. Any FAIL fails.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use pil2_pilout::pilout::PilOut;
use pil2_pilout::pilout_proxy::PilOutProxy;
use pil2_stark_recurser::plonk2pil::check::load::{check_result, template};
use pil2_stark_recurser::plonk2pil::field::R1csPrime;
use pil2_stark_recurser::plonk2pil::merge_copies::r1cs2plonk_merged;
use pil2_stark_recurser::plonk2pil::r1cs::types::{r1cs_prime, read_r1cs_from_bytes, read_r1cs_header, PlonkOptions};
use pil2_stark_recurser::plonk2pil::{plonk2pil, FixedValue, PlonkResult, SetupKind};
use proofman_common::hash_family::HashFamily;
use proofman_fields::{Bn128, Goldilocks, PrimeField};
use serde_json::Value;
use sha2::{Digest, Sha256};

const SIZE_HEADER: &str = "case\tfamily\tkind\tnBits\tnBitsNatural\tNUsed\tS\tfixed\tplonk\tn_vars";

/// A fixed-column value, as bytes.
trait Word {
    fn word(&self) -> Vec<u8>;
}
impl Word for u64 {
    fn word(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}
impl Word for Bn128 {
    fn word(&self) -> Vec<u8> {
        self.as_canonical_biguint().to_bytes_le()
    }
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().fold(String::new(), |mut s, b| {
        write!(s, "{b:02x}").unwrap();
        s
    })
}

/// Connection columns `S[j]`: the plonk wires. The committed `a[..]` of the PIL template can be wider
/// (the PIL compiler counts those); the template text is hashed as `_templates`.
fn s_cols(pols: &[pil2_stark_recurser::plonk2pil::FixedPol<impl Sized>]) -> usize {
    pols.iter().filter(|p| p.name.ends_with(".S")).count()
}

struct Row {
    sizes: String,
    hashes: Vec<(String, String)>,
    /// The checker's verdict, with `PCHECK_BUILDS`.
    check: Option<Result<String, String>>,
}

/// A case of `GOLDEN_DIR`.
struct Case {
    name: String,
    family: String,
    kind: SetupKind,
    opts: PlonkOptions,
    bytes: Vec<u8>,
}

fn load_case(dir: &Path) -> Case {
    let name = dir.file_name().unwrap().to_string_lossy().to_string();
    let bytes = std::fs::read(dir.join("circuit.r1cs")).unwrap();
    let meta: Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let s = |k: &str| meta.get(k).and_then(Value::as_str).map(str::to_string);
    let u = |k: &str| meta.get(k).and_then(Value::as_u64).map(|v| v as usize);
    let family = s("family").unwrap();
    let kind = match s("kind").unwrap().as_str() {
        "compressor" => SetupKind::Compressor,
        "aggregation" => SetupKind::Aggregation,
        "wrap" => SetupKind::Wrap,
        other => panic!("{name}: unknown kind {other}"),
    };
    let opts = PlonkOptions {
        hash: family.parse::<HashFamily>().unwrap_or_else(|e| panic!("{name}: {e}")),
        // The default name embeds the time; the pipeline always passes one.
        airgroup_name: s("airgroup").or(Some(name.replace('-', "_"))),
        max_constraint_degree: u("max_degree"),
        blake3_lanes: u("lanes"),
        min_n_bits: u("min_n_bits"),
        ..Default::default()
    };
    Case { name, family, kind, opts, bytes }
}

/// The pilout and the PIL a case's setup compiled, under `PCHECK_BUILDS`.
fn build_of(builds: &Path, case: &str) -> (PathBuf, PathBuf) {
    let (h, rest) = case.split_once('-').expect("<hash>-<name>");
    if let Some(kind) = rest.strip_prefix("rt-") {
        let d = builds.join(format!("rt-{h}-{kind}"));
        return (d.join("build/Compressor.pilout"), d.join("pil/Compressor.pil"));
    }
    let stem = match rest {
        "recursive2" => "FiboCPU_recursive2",
        "compressor" => "FibonacciSquare_compressor",
        "wrap" => "final",
        s => s,
    };
    let d = builds.join(format!("b-{h}"));
    (d.join(format!("build/{stem}.pilout")), d.join(format!("pil/{stem}.pil")))
}

fn pilout(path: &Path) -> PilOut {
    PilOutProxy::new(path.to_str().unwrap()).unwrap_or_else(|e| panic!("{}: {e}", path.display())).pilout
}

/// The checker on `r` against its setup's pilout; `Err("STALE")` if that setup compiled another PIL.
fn check<V: FixedValue>(c: &Case, r: &PlonkResult<V>, builds: &Path) -> Result<String, String> {
    let (po, pil) = build_of(builds, &c.name);
    let disk = std::fs::read_to_string(&pil).map_err(|e| format!("{}: {e}", pil.display()))?;
    if template(&disk).ok() != template(&r.pil_str).ok() {
        return Err("STALE".into());
    }
    check_result(&c.bytes, r, &pilout(&po)).map(|rep| rep.to_string()).map_err(|e| format!("{e:#}"))
}

fn run<V: FixedValue + Word>(c: &Case, plonk: usize, n_vars: u32, builds: Option<&Path>) -> Row {
    let name = &c.name;
    let r: PlonkResult<V> =
        plonk2pil(&c.bytes, c.kind, &c.opts).unwrap_or_else(|e| panic!("{name}: plonk2pil failed: {e:#}"));
    let committed = s_cols(&r.fixed_pols);
    let sizes = format!(
        "{name}\t{}\t{}\t{}\t{}\t{}\t{committed}\t{}\t{plonk}\t{n_vars}",
        c.family,
        c.kind,
        r.n_bits,
        r.n_bits_natural,
        r.n_used,
        r.fixed_pols.len()
    );
    let mut hashes = vec![
        ("exec".to_string(), sha(&r.exec.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>())),
        ("pil_str".to_string(), sha(r.pil_str.as_bytes())),
    ];
    for p in &r.fixed_pols {
        let mut h = Sha256::new();
        for v in &p.values {
            h.update(v.word());
        }
        hashes.push((format!("fixed:{}:{}", p.index, p.name), format!("{:x}", h.finalize())));
    }
    Row { sizes, hashes, check: builds.map(|b| check(c, &r, b)) }
}

fn case(dir: &Path, builds: Option<&Path>) -> Row {
    let c = load_case(dir);
    // The plonk-gate count of the merged r1cs, which no result carries.
    let header = read_r1cs_header(&c.bytes).unwrap();
    match r1cs_prime(&header).unwrap() {
        R1csPrime::Goldilocks => {
            let r1cs = read_r1cs_from_bytes::<Goldilocks>(&c.bytes).unwrap();
            let plonk = r1cs2plonk_merged(&r1cs, c.opts.merge_copies).0.len();
            run::<u64>(&c, plonk, header.n_vars, builds)
        }
        R1csPrime::Bn128 => {
            let r1cs = read_r1cs_from_bytes::<Bn128>(&c.bytes).unwrap();
            let plonk = r1cs2plonk_merged(&r1cs, c.opts.merge_copies).0.len();
            run::<Bn128>(&c, plonk, header.n_vars, builds)
        }
    }
}

fn pil_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            pil_files(&p, out)
        } else if p.extension().is_some_and(|x| x == "pil") {
            out.push(p)
        }
    }
}

fn parse_sizes(s: &str) -> BTreeMap<String, Vec<String>> {
    s.lines()
        .skip(1)
        .map(|l| l.split('\t').map(str::to_string).collect::<Vec<_>>())
        .map(|f| (f[0].clone(), f))
        .collect()
}

fn parse_hashes(s: &str) -> BTreeMap<String, Vec<(String, String)>> {
    let mut m: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for l in s.lines() {
        let f: Vec<_> = l.split('\t').collect();
        m.entry(f[0].to_string()).or_default().push((f[1].to_string(), f[2].to_string()));
    }
    m
}

/// First difference of `case` against the baseline, if any.
fn diff(
    new: &[String],
    old: &[String],
    nh: &[(String, String)],
    oh: &[(String, String)],
    sizes_only: bool,
) -> Option<String> {
    // columns: 3 nBits, 4 natural, 5 NUsed, 6 S, 7 fixed, 8 plonk
    let n = |v: &[String], i: usize| v[i].parse::<usize>().unwrap();
    if n(new, 3) != n(old, 3) {
        return Some(format!("nBits {} -> {}", old[3], new[3]));
    }
    if sizes_only {
        for (i, what) in [(5, "NUsed"), (6, "S"), (8, "plonk")] {
            if n(new, i) > n(old, i) {
                return Some(format!("{what} grew {} -> {}", old[i], new[i]));
            }
        }
        return None;
    }
    if let Some(i) = (4..new.len()).find(|&i| new[i] != old[i]) {
        return Some(format!("size column {i} {} -> {}", old[i], new[i]));
    }
    let (nm, om): (BTreeMap<_, _>, BTreeMap<_, _>) = (nh.iter().cloned().collect(), oh.iter().cloned().collect());
    let first = om.keys().chain(nm.keys()).find(|k| nm.get(*k) != om.get(*k));
    first.cloned()
}

#[test]
#[ignore = "needs GOLDEN_DIR (r1cs cases) and GOLDEN_OUT; see the module doc"]
fn golden() {
    let dir = PathBuf::from(std::env::var("GOLDEN_DIR").expect("GOLDEN_DIR"));
    let out = PathBuf::from(std::env::var("GOLDEN_OUT").expect("GOLDEN_OUT"));
    let sizes_only = std::env::var("GOLDEN_SIZES_ONLY").is_ok();
    let builds = std::env::var("PCHECK_BUILDS").ok().map(PathBuf::from);
    let only = std::env::var("PCHECK_ONLY").unwrap_or_default();
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("circuit.r1cs").exists() && p.to_string_lossy().contains(&only))
        .collect();
    cases.sort();
    assert!(!cases.is_empty(), "no cases in {}", dir.display());

    let (mut sizes, mut hashes, mut check_fails) = (String::from(SIZE_HEADER) + "\n", String::new(), 0);
    for c in &cases {
        let row = case(c, builds.as_deref());
        let name = c.file_name().unwrap().to_string_lossy();
        eprintln!("{}", row.sizes);
        match &row.check {
            None => {}
            Some(Ok(rep)) => println!("CHECK OK {name} {rep}"),
            Some(Err(e)) if e == "STALE" => println!("CHECK STALE {name} the setup's PIL is not this result's"),
            Some(Err(e)) => {
                check_fails += 1;
                println!("CHECK FAIL {name} {e}");
            }
        }
        sizes += &(row.sizes + "\n");
        for (k, h) in row.hashes {
            hashes += &format!("{name}\t{k}\t{h}\n");
        }
    }
    // The PIL templates the pil_str includes: informational (relayouts may change them on purpose).
    let mut tpl = Vec::new();
    pil_files(&Path::new(env!("CARGO_MANIFEST_DIR")).join("plonk2pil/pil"), &mut tpl);
    tpl.sort();
    for f in &tpl {
        let rel = f.strip_prefix(Path::new(env!("CARGO_MANIFEST_DIR")).join("plonk2pil/pil")).unwrap();
        hashes += &format!("_templates\t{}\t{}\n", rel.display(), sha(&std::fs::read(f).unwrap()));
    }
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("sizes.tsv"), &sizes).unwrap();
    std::fs::write(out.join("hashes.tsv"), &hashes).unwrap();
    assert_eq!(check_fails, 0, "the checker rejected {check_fails} case(s)");

    let Ok(base) = std::env::var("GOLDEN_BASELINE") else { return };
    let base = PathBuf::from(base);
    let (os, oh) = (
        parse_sizes(&std::fs::read_to_string(base.join("sizes.tsv")).unwrap()),
        parse_hashes(&std::fs::read_to_string(base.join("hashes.tsv")).unwrap()),
    );
    let (ns, nh) = (parse_sizes(&sizes), parse_hashes(&hashes));
    if nh.get("_templates") != oh.get("_templates") {
        println!("NOTE the PIL templates differ from the baseline's (pil_str hashes only name them)");
    }
    let mut fails = 0;
    for (name, old) in &os {
        let verdict = match ns.get(name) {
            None => Some("missing from this run".to_string()),
            Some(new) => diff(new, old, &nh[name], &oh[name], sizes_only),
        };
        match verdict {
            None => println!("OK {} {} {}", old[1], old[2], name),
            Some(d) => {
                fails += 1;
                println!("FAIL {} {} {name} {d}", old[1], old[2]);
            }
        }
    }
    assert_eq!(fails, 0, "{fails} case(s) differ from the baseline");
}

/// Each tampering of one real case's artefacts must be rejected (`PCHECK_NEG`, default a Poseidon2
/// compressed final: plonk, CMul, FFT4 and Poseidon bands).
#[test]
#[ignore = "needs GOLDEN_DIR and PCHECK_BUILDS; see the module doc"]
fn tampered_artefacts_are_rejected() {
    let dir = PathBuf::from(std::env::var("GOLDEN_DIR").expect("GOLDEN_DIR"));
    let builds = PathBuf::from(std::env::var("PCHECK_BUILDS").expect("PCHECK_BUILDS"));
    let c = load_case(&dir.join(std::env::var("PCHECK_NEG").unwrap_or("Poseidon2-vadcop_final_compressed".into())));
    let res: PlonkResult<u64> = plonk2pil(&c.bytes, c.kind, &c.opts).unwrap();
    let pilout = pilout(&build_of(&builds, &c.name).0);
    check_result(&c.bytes, &res, &pilout).expect("the untampered case is accepted");

    let cs = pil_info::core::from_pilout::from_pilout(
        &pilout,
        0,
        0,
        &pil_info::cfg::FieldCfg::goldilocks(),
        pil_info::cfg::Opening::Deep,
    )
    .unwrap();
    let fid = |n: &str| pil2_stark_recurser::plonk2pil::check::pil::fixed_id(&cs, n, None).unwrap() as usize;
    let inline = |n: &str| -> Vec<bool> {
        pilout.air_groups[0].airs[0].fixed_cols[fid(n)].values.iter().map(|b| b.iter().any(|x| *x != 0)).collect()
    };
    let first = |n: &str| inline(n).iter().position(|x| *x).unwrap();
    let col = |r: &PlonkResult<u64>, n: &str, i: usize| {
        r.fixed_pols.iter().position(|p| p.name.ends_with(&format!(".{n}")) && p.index == i).unwrap()
    };
    let plonk_rows: Vec<usize> = inline("PLONK").iter().enumerate().filter(|(_, x)| **x).map(|(r, _)| r).collect();
    let row = plonk_rows[plonk_rows.len() / 2];
    let reject = |what: &str, r: &PlonkResult<u64>, p: &PilOut| {
        let e = check_result(&c.bytes, r, p).expect_err(what);
        println!("REJECTED {what}: {e:#}");
    };
    let with_fixed = |n: &str, f: &dyn Fn(&mut Vec<Vec<u8>>)| {
        let mut p = pilout.clone();
        f(&mut p.air_groups[0].airs[0].fixed_cols[fid(n)].values);
        p
    };

    let mut r = res.clone();
    let k = col(&r, "C", 1);
    r.fixed_pols[k].values[row] = r.fixed_pols[k].values[row].wrapping_add(1) % 0xFFFF_FFFF_0000_0001;
    reject("a changed plonk coefficient", &r, &pilout);

    // cell (0, i) cut out of its cycle: its predecessor takes its successor
    let mut r = res.clone();
    let ids: Vec<u64> = pilout.air_groups[0].airs[0].fixed_cols[fid("ID")]
        .values
        .iter()
        .map(|b| b.iter().fold(0u64, |a, x| (a << 8) | *x as u64))
        .collect();
    let s0 = col(&r, "S", 0);
    let i = (0..ids.len()).find(|&i| r.fixed_pols[s0].values[i] != ids[i]).unwrap();
    let (pc, pr) = r
        .fixed_pols
        .iter()
        .enumerate()
        .filter(|(_, p)| p.name.ends_with(".S"))
        .find_map(|(c, p)| p.values.iter().position(|v| *v == ids[i]).map(|row| (c, row)))
        .unwrap();
    let succ = r.fixed_pols[s0].values[i];
    r.fixed_pols[s0].values[i] = ids[i];
    r.fixed_pols[pc].values[pr] = succ;
    reject("a cell cut out of its copy cycle", &r, &pilout);

    reject("a plonk selector zeroed", &res, &with_fixed("PLONK", &|v| v[row] = vec![0]));

    let fft = first("FFT4");
    let mut r = res.clone();
    let k = (0..9).map(|i| col(&r, "C", i)).find(|&k| r.fixed_pols[k].values[fft] != 0).unwrap();
    r.fixed_pols[k].values[fft] = 7;
    reject("an FFT4 constant changed", &r, &pilout);

    let at = first("POSEIDONCOMPRESSION");
    reject("a Poseidon anchor moved", &res, &with_fixed("POSEIDONCOMPRESSION", &|v| v.swap(at, at + 1)));
    let at = first("PUBLICS_ROW");
    reject("the publics row zeroed", &res, &with_fixed("PUBLICS_ROW", &|v| v[at] = vec![0]));

    for (what, file) in [("a plonk constraint", "plonk.pil"), ("a CMul constraint", "cmul.pil")] {
        let mut p = pilout.clone();
        let k = cs.constraints.iter().position(|c| c.label.contains(file)).unwrap();
        p.air_groups[0].airs[0].constraints.remove(k);
        reject(&format!("{what} dropped from the PIL"), &res, &p);
    }
}
