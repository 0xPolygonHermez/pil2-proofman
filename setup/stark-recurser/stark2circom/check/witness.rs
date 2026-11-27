//! A check main compiled with the setup's circom and libraries, its witness on an input, and the r1cs it holds.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, ensure, Context, Result};
use proofman_fields::{CubicExtensionField, Goldilocks};

use crate::plonk2pil::field::PlonkField;
use crate::plonk2pil::r1cs::types::{read_r1cs_from_bytes, LinearCombination, R1csFile};

/// The circom binary and `-l` paths the circuit itself compiles with.
pub struct Circom<'a> {
    pub exec: &'a str,
    pub libs: &'a [String],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// `--O1`, as every circuit compiles, and the wasm witness: every `===` holds or the witness fails.
    Wasm,
    /// `--O0` and a C++ witness that does not stop at a failed `===`: each constraint stays its own r1cs
    /// constraint, so the ones a random input breaks can be read off.
    Cpp,
}

pub struct Compiled {
    dir: PathBuf,
    stem: String,
    backend: Backend,
    /// Each main signal's wire (`main.x[i]`), from the `.sym`; a signal the simplification replaced has none.
    wires: HashMap<String, usize>,
}

/// The last few kB of a tool's output: what names its error.
fn tail(s: &[u8]) -> String {
    let s = String::from_utf8_lossy(s);
    let from = (s.len().saturating_sub(4000)..=s.len()).find(|&i| s.is_char_boundary(i)).unwrap_or(0);
    s[from..].to_string()
}

fn run(cmd: &mut Command, what: &str) -> Result<()> {
    let out = cmd.output().with_context(|| format!("running {what}"))?;
    if !out.status.success() {
        bail!("{what} failed:\n{}{}", tail(&out.stdout), tail(&out.stderr));
    }
    Ok(())
}

/// Compiles `src` as `<dir>/<stem>.circom` over `prime` (`goldilocks` or `bn128`).
pub fn compile(c: &Circom, dir: &Path, stem: &str, src: &str, prime: &str, backend: Backend) -> Result<Compiled> {
    fs::create_dir_all(dir)?;
    let file = dir.join(format!("{stem}.circom"));
    fs::write(&file, src)?;
    let mut cmd = Command::new(c.exec);
    cmd.args(["--r1cs", "--sym", "--prime", prime]);
    match backend {
        Backend::Wasm => cmd.args(["--O1", "--wasm"]),
        Backend::Cpp => cmd.args(["--O0", "--c", "--sanity_check", "0"]),
    };
    for l in c.libs {
        cmd.args(["-l", l]);
    }
    run(cmd.arg(&file).arg("-o").arg(dir), &format!("circom on {}", file.display()))?;
    if backend == Backend::Cpp {
        let cpp = dir.join(format!("{stem}_cpp"));
        let sources = ["main.cpp".to_string(), "calcwit.cpp".to_string(), format!("{stem}.cpp")];
        run(
            Command::new("g++")
                .current_dir(&cpp)
                .args(["-std=c++17", "-O1", "-I."])
                .args(sources)
                .args(["-o", stem, "-lgmp"]),
            &format!("g++ on {}", cpp.display()),
        )?;
    }
    let sym = fs::read_to_string(dir.join(format!("{stem}.sym")))?;
    let mut wires = HashMap::new();
    for line in sym.lines() {
        let mut f = line.splitn(4, ',');
        let (wire, name) = (f.nth(1), f.nth(1));
        let (Some(wire), Some(name)) = (wire, name) else { bail!("a .sym line `{line}`") };
        let wire: i64 = wire.parse().with_context(|| format!("a .sym line `{line}`"))?;
        if let (Some(name), Ok(wire)) = (name.strip_prefix("main."), usize::try_from(wire)) {
            if !name.contains('.') {
                wires.insert(name.to_string(), wire);
            }
        }
    }
    Ok(Compiled { dir: dir.to_path_buf(), stem: stem.to_string(), backend, wires })
}

impl Compiled {
    /// The wire of the main's signal `name` (`roots[0][2]`), if the simplification kept it.
    pub fn wire(&self, name: &str) -> Option<usize> {
        self.wires.get(name).copied()
    }

    pub fn r1cs<F: PlonkField>(&self) -> Result<R1csFile<F>> {
        let p = self.dir.join(format!("{}.r1cs", self.stem));
        read_r1cs_from_bytes(&fs::read(&p).with_context(|| p.display().to_string())?)
    }

    /// The witness on `input` (a main's input JSON), as `<dir>/<stem>_<tag>.wtns`.
    pub fn witness<F: PlonkField>(&self, input: &str, tag: &str) -> Result<Vec<F>> {
        let (json, wtns) =
            (self.dir.join(format!("{}_{tag}.json", self.stem)), self.dir.join(format!("{}_{tag}.wtns", self.stem)));
        fs::write(&json, input)?;
        let stem = &self.stem;
        match self.backend {
            Backend::Wasm => {
                let js = self.dir.join(format!("{stem}_js"));
                run(
                    Command::new("node")
                        .arg(js.join("generate_witness.js"))
                        .arg(js.join(format!("{stem}.wasm")))
                        .arg(&json)
                        .arg(&wtns),
                    "the wasm witness",
                )?
            }
            Backend::Cpp => {
                run(Command::new(self.dir.join(format!("{stem}_cpp/{stem}"))).arg(&json).arg(&wtns), "the C++ witness")?
            }
        }
        read_wtns(&fs::read(&wtns)?)
    }
}

/// The values of a `.wtns` file: the magic `wtns`, a version and a section count, then sections of
/// `(type: u32, size: u64, data)`; section 1 holds `n8`, the prime and the count, section 2 the values.
pub fn read_wtns<F: PlonkField>(data: &[u8]) -> Result<Vec<F>> {
    let u32_at = |at: usize| -> Result<usize> {
        Ok(u32::from_le_bytes(data.get(at..at + 4).context("short .wtns")?.try_into()?) as usize)
    };
    ensure!(data.get(..4) == Some(&b"wtns"[..]), "not a .wtns file");
    let (mut header, mut values) = (None, None);
    let mut at = 12;
    for _ in 0..u32_at(8)? {
        let kind = u32_at(at)?;
        let size = u64::from_le_bytes(data.get(at + 4..at + 12).context("short .wtns")?.try_into()?) as usize;
        let body = data.get(at + 12..at + 12 + size).context("short .wtns")?;
        match kind {
            1 => header = Some(body),
            2 => values = Some(body),
            _ => {}
        }
        at += 12 + size;
    }
    let (header, values) = (header.context(".wtns without a header")?, values.context(".wtns without values")?);
    let n8 = u32::from_le_bytes(header.get(..4).context("short .wtns header")?.try_into()?) as usize;
    ensure!(header.get(4..4 + n8) == Some(&F::PRIME.modulus_le()[..]), "the witness is not over {}", F::PRIME);
    let n = u32::from_le_bytes(header.get(4 + n8..8 + n8).context("short .wtns header")?.try_into()?) as usize;
    ensure!(values.len() == n * n8, ".wtns: {} bytes of values for {n}", values.len());
    values.chunks_exact(n8).map(|v| F::from_canonical_le(v).context("a non-canonical witness value")).collect()
}

fn eval<F: PlonkField>(lc: &LinearCombination<F>, w: &[F]) -> F {
    lc.iter().fold(F::ZERO, |acc, (&wire, &c)| acc + c * w[wire as usize])
}

/// The constraints `w` does not satisfy, by index.
pub fn unsatisfied<F: PlonkField>(r1cs: &R1csFile<F>, w: &[F]) -> Result<Vec<usize>> {
    ensure!(w.len() == r1cs.header.n_vars as usize, "{} witness values for {} wires", w.len(), r1cs.header.n_vars);
    Ok((r1cs.constraints.iter().enumerate())
        .filter(|(_, c)| eval(&c.a, w) * eval(&c.b, w) != eval(&c.c, w))
        .map(|(i, _)| i)
        .collect())
}

/// Every custom gate use is a `CMul` its signals satisfy: `out = ina * inb` in the cubic extension, the
/// relation plonk2pil's CMul gate enforces (signals `ina[3]`, `inb[3]`, `out[3]`).
pub fn check_cmul_gates(r1cs: &R1csFile<Goldilocks>, w: &[Goldilocks]) -> Result<()> {
    for (u, g) in r1cs.custom_gates_uses.iter().enumerate() {
        let gate = r1cs.custom_gates.get(g.id as usize).context("a custom gate use of no gate")?;
        ensure!(gate.template_name == "CMul", "custom gate {} in a checked template", gate.template_name);
        let s = (g.signals.iter())
            .map(|&i| w.get(i as usize).copied().context("a CMul signal past the witness"))
            .collect::<Result<Vec<_>>>()?;
        ensure!(s.len() == 9, "a CMul of {} signals", s.len());
        let e = |i: usize| CubicExtensionField { value: [s[i], s[i + 1], s[i + 2]] };
        if (e(0) * e(3)).value != e(6).value {
            bail!("CMul use {u} does not hold: its witness is not ina * inb");
        }
    }
    Ok(())
}
