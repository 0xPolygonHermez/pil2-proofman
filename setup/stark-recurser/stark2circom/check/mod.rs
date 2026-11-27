//! Trusted checker of the circom the setup generates: the constraint templates rendered from the constraint DAG
//! (`ConstraintRoots<sfx>` and its parts per GL key, BN128 `ConstraintEval`, vadcop_final's `GlobalConstraints`)
//! and the main's dispatch of its keys to them.
//!
//! Inputs are artefacts only: the generated files, each key's starkinfo and verifierinfo, the globalInfo and how
//! the circuit uses its keys; never the generator's structures. Per template:
//! - its text only declares signals and constrains them (`<==`, `===`), [`syntax::constraint_only`];
//! - compiled alone in a main shaped as the static library calls it, with the circuit's circom and libraries, on
//!   two random inputs from the OS: every r1cs constraint (and CMul gate) holds on the witness, and each root it
//!   outputs is the DAG's, evaluated by the `constraint-dag` crate ([`values`]). `GlobalConstraints` has no
//!   outputs: compiled at `--O0`, the constraints a random input breaks must be exactly one zero check per
//!   non-zero limb of each root.
//!
//! Trusted: circom, the static library, the wasm and C++ witness generators, `constraint-dag`.

pub mod syntax;
pub mod values;
pub mod witness;
#[cfg(test)]
mod tests;

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use num_bigint::BigUint;
use proofman_common::StarkInfo;
use proofman_fields::{Bn128, Goldilocks, PrimeField, PrimeField64, QuotientMap};

use crate::plonk2pil::field::PlonkField;
use crate::plonk2pil::r1cs::types::R1csFile;
use serde_json::Value;

use syntax::Block;
use values::{limbs, Air, Global, Rng, Signal};
use witness::{Backend, Circom, Compiled};

/// Random inputs each template is checked on.
const ROUNDS: usize = 2;

/// The field a circuit is over: GL keys render `ConstraintRoots`, a BN128 key `ConstraintEval`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Field {
    Goldilocks,
    Bn128,
}

/// One key a circuit verifies: its verifier circom and the artefacts the setup made it from.
pub struct Key<'a> {
    pub verifier_file: &'a Path,
    pub stark_info: &'a Value,
    pub verifier_info: &'a Value,
    pub const_root: Option<&'a [String; 4]>,
    pub enable_input: bool,
    pub input_challenges: bool,
}

/// A generated circuit: its main's text, its keys in the main's order, its globalInfo (`Null` if none).
pub struct Circuit<'a> {
    pub field: Field,
    pub main: &'a str,
    pub keys: Vec<Key<'a>>,
    pub global_info: &'a Value,
    /// It checks the global constraints (vadcop_final).
    pub global_constraints: bool,
}

/// Checks `circuit`, compiling in `work_dir`; how long each part took.
pub fn check_circuit(circom: &Circom, work_dir: &Path, circuit: &Circuit) -> Result<Vec<(String, Duration)>> {
    let mut times = Vec::new();
    let suffixes = match circuit.field {
        Field::Goldilocks if !circuit.keys.is_empty() => syntax::roots_dispatch(circuit.main, circuit.keys.len())?,
        _ => vec![String::new(); circuit.keys.len()],
    };
    for (k, (key, suffix)) in circuit.keys.iter().zip(&suffixes).enumerate() {
        let t = Instant::now();
        let name = key.verifier_file.file_name().map_or("?".into(), |n| n.to_string_lossy().into_owned());
        check_key(circom, &work_dir.join(format!("key{k}")), circuit, k, suffix)
            .with_context(|| format!("key {k} ({name})"))?;
        times.push((name, t.elapsed()));
    }
    if circuit.global_constraints {
        let t = Instant::now();
        check_global(circom, &work_dir.join("global"), circuit.main, circuit.global_info)
            .context("GlobalConstraints")?;
        times.push(("GlobalConstraints".into(), t.elapsed()));
    }
    Ok(times)
}

/// The generated templates of a verifier file, which holds nothing else besides its pragmas and includes.
fn generated_blocks<'a>(src: &'a str, allowed: impl Fn(&str) -> bool) -> Result<Vec<Block<'a>>> {
    let blocks = syntax::template_blocks(src)?;
    let mut rest = src.to_string();
    for b in &blocks {
        ensure!(allowed(b.name), "a template {} besides the constraint ones", b.name);
        syntax::constraint_only(b)?;
        rest = rest.replacen(b.text, "", 1);
    }
    for l in rest.lines().map(str::trim) {
        let header = l.starts_with("pragma ") || l.starts_with("include \"");
        ensure!(l.is_empty() || l.starts_with("//") || header, "a line outside the templates: `{l}`");
    }
    Ok(blocks)
}

fn is_part_of(name: &str, base: &str) -> bool {
    name == base
        || name.strip_prefix(base).and_then(|r| r.strip_prefix("Part")).is_some_and(|k| k.parse::<u32>().is_ok())
}

/// The library files the generated templates call into, and the blocks themselves: a program of only these.
fn program(includes: &[&str], blocks: &[Block]) -> String {
    let includes: String = includes.iter().map(|f| format!("include \"{f}\";\n")).collect();
    let templates: Vec<&str> = blocks.iter().map(|b| b.text).collect();
    format!("pragma circom 2.1.0;\npragma custom_templates;\n{includes}\n{}\n", templates.join("\n"))
}

fn check_key(circom: &Circom, dir: &Path, circuit: &Circuit, k: usize, suffix: &str) -> Result<()> {
    let (key, main) = (&circuit.keys[k], circuit.main);
    let si = StarkInfo::try_from_value(key.stark_info)?;
    let air = Air::new(&si, key.verifier_info)?;

    let src = fs::read_to_string(key.verifier_file).with_context(|| key.verifier_file.display().to_string())?;
    let mut rng = Rng::new()?;
    match circuit.field {
        Field::Goldilocks => {
            let name = format!("ConstraintRoots{suffix}");
            let blocks = generated_blocks(&src, |b| is_part_of(b, &name))?;
            ensure!(blocks.iter().any(|b| b.name == name), "no template {name}");
            let inputs = air.random_inputs(&mut rng)?;
            let main = format!(
                "{}template CheckedRoots() {{\n{}    signal output roots[{}][3];\n    \
                 roots <== {name}()(evals, challenges, publics, proofvalues, airvalues, airgroupvalues);\n}}\n\ncomponent main = CheckedRoots();\n",
                program(&["cmul.circom"], &blocks),
                declarations(&inputs),
                air.n_roots
            );
            let compiled = witness::compile(circom, dir, "roots", &main, "goldilocks", Backend::Wasm)?;
            let r1cs = compiled.r1cs::<Goldilocks>()?;
            for round in 0..ROUNDS {
                let inputs = if round == 0 { inputs.clone() } else { air.random_inputs(&mut rng)? };
                let w = holding_witness(&compiled, &r1cs, &inputs, round)?;
                witness::check_cmul_gates(&r1cs, &w)?;
                for (j, root) in air.roots(&inputs)?.iter().enumerate() {
                    let got = (0..3)
                        .map(|l| Ok(output(&compiled, &w, &format!("roots[{j}][{l}]"))?.as_canonical_u64()))
                        .collect::<Result<Vec<u64>>>()?;
                    if got != limbs(root) {
                        bail!("root {j} ({}) is {got:?}, the DAG gives {:?}", air.labels[j], limbs(root));
                    }
                }
            }
        }
        Field::Bn128 => {
            let blocks = generated_blocks(&src, |b| b == "ConstraintEval")?;
            // the family file the verifier includes defines the GL ops and the verifier, which reads the key data
            let includes: Vec<&str> =
                src.lines().filter_map(|l| l.strip_prefix("include \"")?.strip_suffix("\";")).collect();
            let key_data: String =
                main.lines().filter(|l| l.starts_with("function key_")).map(|l| format!("{l}\n")).collect();
            let program = program(&includes, &blocks) + &key_data;
            let inputs = air.random_bn128_inputs(&mut rng)?;
            let main = bn128_main(&program, &inputs);
            let compiled = witness::compile(circom, dir, "eval", &main, "bn128", Backend::Wasm)?;
            let r1cs = compiled.r1cs::<Bn128>()?;
            for round in 0..ROUNDS {
                let inputs = if round == 0 { inputs.clone() } else { air.random_bn128_inputs(&mut rng)? };
                let w = holding_witness(&compiled, &r1cs, &inputs, round)?;
                // a lazily reduced Goldilocks value
                let p = BigUint::from(Goldilocks::ORDER_U64);
                let out = (0..3)
                    .map(|l| {
                        let x = output(&compiled, &w, &format!("evalP[{l}]"))?.as_canonical_biguint() % &p;
                        Ok(u64::try_from(x).expect("below the Goldilocks prime"))
                    })
                    .collect::<Result<Vec<u64>>>()?;
                let want = limbs(&air.folded_times_zh(&inputs)?);
                ensure!(out == want, "evalP is {out:?}, the DAG's roots folded times Zh are {want:?}");
            }
        }
    }
    fs::remove_dir_all(dir).ok();
    Ok(())
}

fn declarations(inputs: &[Signal]) -> String {
    inputs.iter().map(Signal::declaration).collect()
}

/// `StarkVerifierBN128`'s call of `ConstraintEval`: Goldilocks words below 2^64, the `maxNum` the verifier gives them.
fn bn128_main(program: &str, inputs: &[Signal]) -> String {
    let mut body = declarations(inputs);
    body += "    signal output evalP[3];\n\n    component ce = ConstraintEval();\n";
    for s in inputs {
        let dims: String = s.dims.iter().map(|d| format!("[{d}]")).collect();
        body += &format!("    signal {{maxNum}} {0}GL{dims};\n    {0}GL.maxNum = max64();\n", s.name);
        if !s.values.is_empty() {
            body += &format!("    {0}GL <== {0};\n", s.name);
        }
        body += &format!("    ce.{0} <== {0}GL;\n", s.name);
    }
    format!(
        "{program}template CheckedEval() {{\n{body}    evalP <== ce.evalP;\n}}\n\ncomponent main = CheckedEval();\n"
    )
}

/// The witness on `inputs`: each input the simplification kept holds its value, every r1cs constraint holds.
fn holding_witness<F: PlonkField + QuotientMap<u64>>(
    compiled: &Compiled,
    r1cs: &R1csFile<F>,
    inputs: &[Signal],
    round: usize,
) -> Result<Vec<F>> {
    let w: Vec<F> = compiled.witness(&values::input_json(inputs), &round.to_string())?;
    check_inputs(compiled, &w, inputs)?;
    let failing = witness::unsatisfied(r1cs, &w)?;
    if let Some(&c) = failing.first() {
        bail!("{} r1cs constraints do not hold on the witness, the first {c}", failing.len());
    }
    Ok(w)
}

fn check_inputs<F: PlonkField + QuotientMap<u64>>(compiled: &Compiled, w: &[F], inputs: &[Signal]) -> Result<()> {
    for s in inputs {
        for (name, &v) in s.entries().iter().zip(&s.values) {
            if let Some(wire) = compiled.wire(name) {
                ensure!(w.get(wire) == Some(&F::from_int(v)), "input {name} is not the value given");
            }
        }
    }
    Ok(())
}

/// The main's output `name` on witness `w`.
fn output<F: Copy>(compiled: &Compiled, w: &[F], name: &str) -> Result<F> {
    compiled.wire(name).and_then(|i| w.get(i).copied()).with_context(|| format!("the main has no output {name}"))
}

/// `GlobalConstraints` of the main, as `VadcopFinal` instantiates it. At `--O0` each `x === 0` stays the r1cs
/// constraint `0 * 0 = c * x`; on random inputs the ones that fail must be one per non-zero limb of each root,
/// with that limb's value, and every other constraint holds.
fn check_global(circom: &Circom, dir: &Path, main: &str, gi: &Value) -> Result<()> {
    let global = Global::new(gi)?;
    let blocks: Vec<Block> =
        syntax::template_blocks(main)?.into_iter().filter(|b| is_part_of(b.name, "GlobalConstraints")).collect();
    ensure!(blocks.iter().any(|b| b.name == "GlobalConstraints"), "the main has no GlobalConstraints");
    for b in &blocks {
        syntax::constraint_only(b)?;
    }
    let mut rng = Rng::new()?;
    let inputs = global.random_inputs(&mut rng)?;
    let params = global.params.map(|p| p.to_string()).join(", ");
    let mut body = declarations(&inputs);
    body += &format!("\n    component g = GlobalConstraints({params});\n");
    for s in &inputs {
        body += &format!("    g.{0} <== {0};\n", s.name);
    }
    let src = format!(
        "{}template CheckedGlobal() {{\n{body}}}\n\ncomponent main = CheckedGlobal();\n",
        program(&["cmul.circom"], &blocks)
    );
    let compiled = witness::compile(circom, dir, "global", &src, "goldilocks", Backend::Cpp)?;
    let r1cs = compiled.r1cs::<Goldilocks>()?;
    for round in 0..ROUNDS {
        let inputs = if round == 0 { inputs.clone() } else { global.random_inputs(&mut rng)? };
        let w: Vec<Goldilocks> = compiled.witness(&values::input_json(&inputs), &round.to_string())?;
        witness::check_cmul_gates(&r1cs, &w)?;
        check_inputs(&compiled, &w, &inputs)?;
        let mut zero_checked: Vec<u64> = Vec::new();
        for c in witness::unsatisfied(&r1cs, &w)? {
            let r = &r1cs.constraints[c];
            let single = r.a.is_empty() && r.b.is_empty() && r.c.len() == 1;
            match r.c.keys().next() {
                Some(&wire) if single && wire != 0 => zero_checked.push(w[wire as usize].as_canonical_u64()),
                _ => bail!("r1cs constraint {c} fails and is not a zero check"),
            }
        }
        let want = global.roots(&inputs)?;
        let mut limbs_of_roots: Vec<(u64, usize)> =
            (want.iter().enumerate()).flat_map(|(j, r)| limbs(r).into_iter().map(move |l| (l, j))).collect();
        limbs_of_roots.retain(|&(l, _)| l != 0);
        zero_checked.sort_unstable();
        for (l, j) in &limbs_of_roots {
            match zero_checked.binary_search(l) {
                Ok(i) => {
                    zero_checked.remove(i);
                }
                Err(_) => bail!("global root {j} is not constrained to zero (a limb is {l})"),
            }
        }
        if let Some(x) = zero_checked.first() {
            bail!("{} zero checks of values no root has (one is {x})", zero_checked.len());
        }
    }
    fs::remove_dir_all(dir).ok();
    Ok(())
}
