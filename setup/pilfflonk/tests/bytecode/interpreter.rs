//! The fixtures of the C++ interpreter's tests (M17, `pil2-stark/test/pilfflonk/
//! pilfflonk_expressions_test.cpp`), in `tests/fixtures/bytecode/`:
//!
//! - `Sample.expected.json`: inputs for the codes of [`sample`] on the trace domain, on the
//!   extended coset and at a point, and the values the Rust evaluator here gives them over `Fr`
//!   (`num-bigint`, through the oracle's `Fr`). It also has the zerofier terms on the coset and at
//!   the point, from the closed forms of A.1 (`Domain::zerofier_at`).
//! - `fibonacci/`: the Fibonacci of M13, `N = 256`. It has what `setup-pilfflonk --no-packing`
//!   writes for it (`Fibonacci.bin`, `.pilfflonkinfo.json`, `.const`), the stage-1 trace of M13's
//!   generator for `[1, 2]` (`Fibonacci.witness.bin`, the `instance_0_0_0.bin` of its witness
//!   directory), the `qVerifier` encoded as a bytecode of one expression (`Fibonacci.qverifier.bin`),
//!   and what the oracle (M14) gives at a point `ξ` (`Fibonacci.oracle.json`): the evaluations of
//!   the evMap, the zerofier terms and `Q(ξ)`.
//! - `sum_bus/`: the lookup on the std's sum bus of plan M30, `N = 32`, of two stages, with the
//!   std's default `MAX_CONSTRAINT_DEGREE` (plan M31). It has what `setup-pilfflonk --no-packing`
//!   writes for it (`SumBus.bin`, whose section 3 has the hints `im_col` and `gsum_col`, which reads
//!   the column of the first; `.pilfflonkinfo.json`, `.const`), the stage-1 traces of its
//!   generator's witness (`SumBus.witness.bin`) and of the one that looks up a value the table does
//!   not provide (`SumBus.broken.bin`), and what the oracle gives for stage 2 with fixed challenges
//!   (`SumBus.oracle.json`): the challenges, `std_alpha` and `std_gamma`, the publics, and every
//!   column of stage 2 by `stagePos`, `gsum` and `im_single` from the pilout's hints and the im pols
//!   from their expressions.
//!
//! The tests below check that the checked-in files are what they compute, and write them with
//! `PILFFLONK_UPDATE_FIXTURES=1`, as the Sample.bin test does. The Fibonacci one needs `PIL2C_EXEC`.

use std::fs;
use std::path::{Path, PathBuf};

use prost::Message;
use serde_json::{json, Value};

use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::passes;
use pilfflonk_setup::test_ptau::write_tau_one_ptau;
use pilfflonk_setup::{run_setup_pilfflonk, SetupPilfflonkOptions};
use proofman_pilfflonk::oracle::{self, omega, AirOracle, ColumnRef, Domain, Fr};
use proofman_pilfflonk::{
    AirFile, Boundary, EvMapEntry, FileWitnessSource, JsonFile, PilfflonkGlobalInfo, PilfflonkInfo, PolType,
};

use super::*;

/// M13's generator, `proofman-pilfflonk`'s test module, included as it is.
#[path = "../../../../pilfflonk/tests/data/fibonacci.rs"]
mod fibonacci;
/// The generator of the sum bus (plan M30).
#[path = "../../../../pilfflonk/tests/data/sum_bus.rs"]
mod sum_bus;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/bytecode");

/// The shift of the extended coset (spec §4.4, M5's `COSET_SHIFT`).
const COSET_SHIFT: u64 = 5;

/// `bytes` is the checked-in fixture `name`, which `PILFFLONK_UPDATE_FIXTURES=1` writes first.
fn check_fixture(name: &str, bytes: &[u8]) {
    let path = Path::new(FIXTURES).join(name);
    if std::env::var_os("PILFFLONK_UPDATE_FIXTURES").is_some() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
    }
    let fixture = fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(fixture == bytes, "{} is not what the test computes: regenerate it (see the module)", path.display());
}

/// The fixture's text, with the object keys sorted explicitly: serde_json keeps insertion order
/// instead when `preserve_order` is on, which Cargo turns on for the whole build whenever a crate
/// that enables it (`setup/stark-recurser`) is part of it.
fn json_text(value: &Value) -> Vec<u8> {
    let mut text = serde_json::to_string_pretty(&sorted(value)).unwrap();
    text.push('\n');
    text.into_bytes()
}

fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            Value::Object(keys.into_iter().map(|k| (k.clone(), sorted(&map[k]))).collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

fn dec(v: &Fr) -> Value {
    Value::String(v.to_string())
}

fn decs(values: &[Fr]) -> Value {
    Value::Array(values.iter().map(dec).collect())
}

fn from_bytes(v: &FrBytes) -> Fr {
    Fr::from(v)
}

// ---------------------------------------------------------------------------------------------
// The Rust evaluator
// ---------------------------------------------------------------------------------------------

/// What the operands of a code block are, on a domain of `m` points (1 for a single point).
#[derive(Default)]
struct Inputs {
    opening_points: Vec<i64>,
    /// `2^extend_bits` points of the domain per row of the trace: 0 on `H` and at a point.
    extend_bits: u32,
    /// `columns[type][arg1][point]`: the fixed columns (type 0) and those of each stage.
    columns: Vec<Vec<Vec<Fr>>>,
    /// `zerofiers[boundary][point]`.
    zerofiers: Vec<Vec<Fr>>,
    publics: Vec<Fr>,
    challenges: Vec<Fr>,
    air_values: Vec<Fr>,
    airgroup_values: Vec<Fr>,
    proof_values: Vec<Fr>,
    evals: Vec<Fr>,
}

impl Inputs {
    fn column(&self, ty: u32, arg1: u32, opening: u32, point: usize) -> Fr {
        let column = &self.columns[ty as usize][arg1 as usize];
        let shift = self.opening_points[opening as usize] << self.extend_bits;
        column[(point as i64 + shift).rem_euclid(column.len() as i64) as usize].clone()
    }

    fn value(&self, t: &Operand, tmps: &[Fr], point: usize) -> Fr {
        match *t {
            Operand::Const { id, opening } => self.column(0, id, opening, point),
            Operand::Cm { stage, stage_pos, opening } => self.column(stage, stage_pos, opening, point),
            Operand::Zi { boundary } => self.zerofiers[boundary as usize][point].clone(),
            Operand::Tmp(slot) => tmps[slot as usize].clone(),
            Operand::Number(v) => from_bytes(&v),
            Operand::Public(i) => self.publics[i as usize].clone(),
            Operand::Challenge(i) => self.challenges[i as usize].clone(),
            Operand::AirValue(i) => self.air_values[i as usize].clone(),
            Operand::AirgroupValue(i) => self.airgroup_values[i as usize].clone(),
            Operand::ProofValue(i) => self.proof_values[i as usize].clone(),
            Operand::Eval(i) => self.evals[i as usize].clone(),
        }
    }

    /// The value of `code` at `point`, as the format's semantics give it.
    fn evaluate(&self, code: &Code, point: usize) -> Fr {
        let mut tmps = vec![Fr::zero(); code.n_temp as usize];
        for op in &code.ops {
            let (a, b) = (self.value(&op.a, &tmps, point), self.value(&op.b, &tmps, point));
            tmps[op.dest as usize] = match op.opcode {
                Opcode::Add => &a + &b,
                Opcode::Sub => &a - &b,
                Opcode::Mul => &a * &b,
                Opcode::SubSwap => &b - &a,
            };
        }
        tmps[code.dest_id as usize].clone()
    }

    /// The values of `code` on every point of a domain of `m` points.
    fn evaluate_all(&self, code: &Code, m: usize) -> Vec<Fr> {
        (0..m).map(|point| self.evaluate(code, point)).collect()
    }
}

fn domain(b: &Boundary) -> Domain {
    match *b {
        Boundary::EveryRow => Domain::EveryRow,
        Boundary::FirstRow => Domain::FirstRow,
        Boundary::LastRow => Domain::LastRow,
        Boundary::EveryFrame { offset_min, offset_max } => {
            Domain::EveryFrame { offset_min: offset_min as usize, offset_max: offset_max as usize }
        }
    }
}

/// `Zi` of `boundary` at `x ∉ H`: `1/Z_H(x)` for everyRow, `Z_H(x)/Z_D(x)` for any other (A.1).
fn zerofier_term(boundary: &Boundary, x: &Fr, n_bits: u32) -> Fr {
    let z_h = &x.pow_u64(1 << n_bits) - &Fr::one();
    match boundary {
        Boundary::EveryRow => z_h.inv().unwrap(),
        other => &z_h * &domain(other).zerofier_at(x, n_bits).unwrap().inv().unwrap(),
    }
}

/// The points of the extended coset `g·H'`, `g·ω_{N'}^i`.
fn coset_points(n_bits_ext: u32) -> Vec<Fr> {
    let (g, w) = (Fr::from_u64(COSET_SHIFT), omega(n_bits_ext).unwrap());
    let mut x = g;
    (0..1usize << n_bits_ext)
        .map(|_| {
            let point = x.clone();
            x = &x * &w;
            point
        })
        .collect()
}

fn boundary_json(b: &Boundary) -> Value {
    match b {
        Boundary::EveryRow => json!({"name": "everyRow"}),
        Boundary::FirstRow => json!({"name": "firstRow"}),
        Boundary::LastRow => json!({"name": "lastRow"}),
        Boundary::EveryFrame { offset_min, offset_max } => {
            json!({"name": "everyFrame", "offsetMin": offset_min, "offsetMax": offset_max})
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Sample.expected.json
// ---------------------------------------------------------------------------------------------

/// Values of 254 bits or so, each different: `(2s + 3)^97`.
struct Values(u64);

impl Values {
    fn next(&mut self) -> Fr {
        self.0 += 1;
        Fr::from_u64(2 * self.0 + 3).pow_u64(97)
    }

    fn take(&mut self, n: usize) -> Vec<Fr> {
        (0..n).map(|_| self.next()).collect()
    }

    /// The columns of [`sample`]'s AIR on `m` points: const0; a, b and im of stage 1; c of stage 2.
    fn columns(&mut self, m: usize) -> Vec<Vec<Vec<Fr>>> {
        vec![vec![self.take(m)], (0..3).map(|_| self.take(m)).collect(), vec![self.take(m)]]
    }
}

/// The AIR [`sample`] is for (see its documentation): `N = 8`, the extended coset of `N' = 16`.
const SAMPLE_N_BITS: u32 = 3;
const SAMPLE_N_BITS_EXT: u32 = 4;

fn sample_boundaries() -> Vec<Boundary> {
    vec![
        Boundary::EveryRow,
        Boundary::FirstRow,
        Boundary::LastRow,
        Boundary::EveryFrame { offset_min: 1, offset_max: 2 },
    ]
}

/// What [`sample`]'s codes give, on the trace domain (the intermediate polynomial 3, the value 9
/// and the constraints), on the extended coset (`Q`, 7) and at a point `ξ` (11).
fn sample_expected() -> Value {
    let bytecode = sample();
    let boundaries = sample_boundaries();
    let (n, m) = (1usize << SAMPLE_N_BITS, 1usize << SAMPLE_N_BITS_EXT);
    let mut values = Values(0);
    let scalars = |values: &mut Values| Inputs {
        publics: values.take(2),
        challenges: values.take(1),
        air_values: values.take(1),
        airgroup_values: values.take(1),
        proof_values: values.take(2),
        ..Default::default()
    };
    let opening_points = vec![-2, -1, 0, 1, 2];

    // The same scalars everywhere; columns of their own on H and on the coset.
    let base = scalars(&mut values);
    let with_scalars = |columns, extend_bits, zerofiers, evals| Inputs {
        opening_points: opening_points.clone(),
        extend_bits,
        columns,
        zerofiers,
        publics: base.publics.clone(),
        challenges: base.challenges.clone(),
        air_values: base.air_values.clone(),
        airgroup_values: base.airgroup_values.clone(),
        proof_values: base.proof_values.clone(),
        evals,
    };
    let trace = with_scalars(values.columns(n), 0, Vec::new(), Vec::new());
    let points = coset_points(SAMPLE_N_BITS_EXT);
    let coset_zerofiers: Vec<Vec<Fr>> =
        boundaries.iter().map(|b| points.iter().map(|x| zerofier_term(b, x, SAMPLE_N_BITS)).collect()).collect();
    let coset = with_scalars(values.columns(m), SAMPLE_N_BITS_EXT - SAMPLE_N_BITS, coset_zerofiers.clone(), Vec::new());
    let xi = values.next();
    let xi_zerofiers: Vec<Fr> = boundaries.iter().map(|b| zerofier_term(b, &xi, SAMPLE_N_BITS)).collect();
    let evals = values.take(4);
    let point = with_scalars(Vec::new(), 0, xi_zerofiers.iter().map(|z| vec![z.clone()]).collect(), evals.clone());

    let expression = |exp_id: u32| &bytecode.expressions.iter().find(|e| e.exp_id == exp_id).unwrap().code;
    let columns_json = |inputs: &Inputs| {
        Value::Array(inputs.columns.iter().map(|cols| Value::Array(cols.iter().map(|c| decs(c)).collect())).collect())
    };
    json!({
        "nBits": SAMPLE_N_BITS,
        "nBitsExt": SAMPLE_N_BITS_EXT,
        "openingPoints": opening_points,
        "boundaries": boundaries.iter().map(boundary_json).collect::<Vec<_>>(),
        "publics": decs(&base.publics),
        "challenges": decs(&base.challenges),
        "airValues": decs(&base.air_values),
        "airgroupValues": decs(&base.airgroup_values),
        "proofValues": decs(&base.proof_values),
        "trace": columns_json(&trace),
        "coset": columns_json(&coset),
        "xi": dec(&xi),
        "evals": decs(&evals),
        "expected": {
            "cosetZerofiers": Value::Array(coset_zerofiers.iter().map(|z| decs(z)).collect()),
            "xiZerofiers": decs(&xi_zerofiers),
            "expressions": {
                "3": decs(&trace.evaluate_all(expression(3), n)),
                "9": decs(&trace.evaluate_all(expression(9), n)),
                "7": decs(&coset.evaluate_all(expression(7), m)),
                "11": dec(&point.evaluate(expression(11), 0)),
            },
            "constraints": Value::Array(
                bytecode.constraints.iter().map(|c| decs(&trace.evaluate_all(&c.code, n))).collect()
            ),
        },
    })
}

#[test]
fn the_cpp_expected_values_are_the_rust_evaluators() {
    check_fixture("Sample.expected.json", &json_text(&sample_expected()));
}

/// Two of the values of `Sample.expected.json` by hand from its inputs, as the comments of
/// [`sample`] write the codes: the evaluator follows them.
#[test]
fn the_rust_evaluator_computes_the_codes_as_written() {
    let expected = sample_expected();
    let fr = |v: &Value| Fr::reduce(&BigUint::parse_bytes(v.as_str().unwrap().as_bytes(), 10).unwrap());
    let scalar = |name: &str, i: usize| fr(&expected[name][i]);

    // Q (7) at point 0 of the coset: ((c·challenge0 + (public1 − airvalue0))·(2^253 + 5) + Zi1·b[+2])·Zi0.
    let (m, e) = (1usize << SAMPLE_N_BITS_EXT, SAMPLE_N_BITS_EXT - SAMPLE_N_BITS);
    let at = |offset: i64| (offset << e).rem_euclid(m as i64) as usize;
    let col = |ty: usize, pos: usize, offset: i64| fr(&expected["coset"][ty][pos][at(offset)]);
    let zi = |b: usize| fr(&expected["expected"]["cosetZerofiers"][b][0]);
    let inner = &(&col(2, 0, 0) * &scalar("challenges", 0)) + &(&scalar("publics", 1) - &scalar("airValues", 0));
    let wide = Fr::from(&FrBytes::from_decimal(WIDE_254).unwrap());
    let q = &(&(&inner * &wide) + &(&zi(1) * &col(1, 1, 2))) * &zi(0);
    assert_eq!(fr(&expected["expected"]["expressions"]["7"][0]), q);

    // The value (9) on any row, a sub_swap among its ops: (airgroupvalue0 − proofvalue1)·public0·(r − 1).
    let r_minus_one = Fr::from(&FrBytes::from_decimal(R_MINUS_ONE).unwrap());
    let v = &(&(&scalar("airgroupValues", 0) - &scalar("proofValues", 1)) * &scalar("publics", 0)) * &r_minus_one;
    assert_eq!(fr(&expected["expected"]["expressions"]["9"][5]), v);
}

// ---------------------------------------------------------------------------------------------
// fibonacci/
// ---------------------------------------------------------------------------------------------

/// A fresh directory of the target's, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("interpreter_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const FIBONACCI_N_BITS: u32 = 8;
const FIBONACCI_INPUTS: [u64; 2] = [1, 2];

/// A point off `H` and `std_vc`, fixed so that the fixture is deterministic: as good as random.
fn xi_and_std_vc() -> (Fr, Fr) {
    (Fr::from_u64(7).pow_u64(1000), Fr::from_u64(0x5eed).pow_u64(77))
}

/// The column an evMap entry evaluates, as the oracle names it.
fn column_of(info: &PilfflonkInfo, e: &EvMapEntry) -> ColumnRef {
    let pol = info.pol(e.pol_type, e.id).unwrap();
    match (e.pol_type, pol.exp_id) {
        (PolType::Const, _) => ColumnRef::Fixed(e.id as usize),
        (PolType::Cm, Some(exp_id)) if pol.im_pol => ColumnRef::Im(exp_id as usize),
        (PolType::Cm, _) => ColumnRef::Witness { stage: pol.stage as usize, idx: pol.stage_id as usize },
    }
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_fibonacci_fixtures_are_the_setups_and_the_oracles() {
    let pilout = compile_bn254("pilfflonk/tests/fixtures/fibonacci/fibonacci.pil");
    let dir = Scratch::new("fibonacci");

    // The setup, as `setup-pilfflonk --no-packing` runs it.
    let opts = SetupPilfflonkOptions {
        airout_path: dir.0.join("fibonacci.pilout"),
        build_dir: dir.0.join("build"),
        powers_of_tau: dir.0.join("tau_one.ptau"),
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: true,
    };
    fs::write(&opts.airout_path, pilout.encode_to_vec()).unwrap();
    write_tau_one_ptau(&opts.powers_of_tau, 1024).unwrap();
    run_setup_pilfflonk(&opts).unwrap();
    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    let gi = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap();
    let air_file = |file| gi.air_file(&proving_key, 0, 0, file).unwrap();
    let info = PilfflonkInfo::read(&air_file(AirFile::PilfflonkInfo)).unwrap();
    let bin = fs::read(air_file(AirFile::Bin)).unwrap();

    // The passes again, as the setup ran them, for the qVerifier.
    let cfg = passes::cfg(DEFAULT_MAX_CONSTRAINT_DEGREE).unwrap();
    let result = pil_info::run(&pilout, 0, 0, &cfg, &Default::default()).unwrap();
    assert_eq!(Bytecode::from_bytes(&bin).unwrap(), Bytecode::from_pil_info(&result).unwrap());
    let context = CodeContext::from_pil_info(&result).unwrap();
    let q_verifier = Code::from_entries(&result.pil_code.verifier_info.q_verifier.code, &context, None).unwrap();
    let q_verifier_bin = Bytecode {
        n_stages: context.n_stages,
        expressions: vec![ExpressionBin {
            exp_id: info.c_exp_id as u32,
            stage: context.n_stages + 1,
            line: "qVerifier".into(),
            code: q_verifier.clone(),
        }],
        constraints: Vec::new(),
        hints: Vec::new(),
    };
    let q_verifier_path = dir.0.join("Fibonacci.qverifier.bin");
    q_verifier_bin.write(&q_verifier_path).unwrap();

    // M13's witness, and the oracle at ξ.
    let witness = fibonacci::witness(FIBONACCI_N_BITS, FIBONACCI_INPUTS);
    let witness_dir = dir.0.join("witness");
    fs::create_dir_all(&witness_dir).unwrap();
    let shape = oracle::witness_shape(&pilout).unwrap();
    witness.write(&witness_dir, &shape).unwrap();
    let air_oracle = AirOracle::new(&pilout, 0, 0).unwrap();
    let source = FileWitnessSource::open(&witness_dir, &shape).unwrap();
    let values = air_oracle.values(&source, 0).unwrap();
    let (xi, std_vc) = xi_and_std_vc();
    let im_pols: Vec<usize> =
        info.cm_pols_map.iter().filter(|p| p.im_pol).map(|p| p.exp_id.unwrap() as usize).collect();
    let q = air_oracle.q_at(&values, &im_pols, &std_vc, &xi).unwrap();
    let evals: Vec<Fr> = info
        .ev_map
        .iter()
        .map(|e| air_oracle.column_at(&values, column_of(&info, e), e.prime as i32, &xi).unwrap())
        .collect();
    let zerofiers: Vec<Fr> = info.boundaries.iter().map(|b| zerofier_term(b, &xi, FIBONACCI_N_BITS)).collect();
    let publics: Vec<Fr> = witness.publics.iter().map(from_bytes).collect();
    let challenges = vec![std_vc.clone(), xi.clone()];
    assert_eq!(info.challenges_map.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["std_vc", "std_xi"]);

    // The qVerifier, in the format and evaluated by the Rust evaluator, is the oracle's Q(ξ).
    let point = Inputs {
        zerofiers: zerofiers.iter().map(|z| vec![z.clone()]).collect(),
        publics: publics.clone(),
        challenges: challenges.clone(),
        evals: evals.clone(),
        ..Default::default()
    };
    assert_eq!(point.evaluate(&q_verifier, 0), q, "the qVerifier at ξ is not the oracle's Q(ξ)");

    let oracle_json = json!({
        "nBits": FIBONACCI_N_BITS,
        "inputs": FIBONACCI_INPUTS,
        "xi": dec(&xi),
        "stdVc": dec(&std_vc),
        "challenges": decs(&challenges),
        "publics": decs(&publics),
        "evals": decs(&evals),
        "zerofiers": decs(&zerofiers),
        "q": dec(&q),
    });
    check_fixture("fibonacci/Fibonacci.bin", &bin);
    check_fixture("fibonacci/Fibonacci.pilfflonkinfo.json", &fs::read(air_file(AirFile::PilfflonkInfo)).unwrap());
    check_fixture("fibonacci/Fibonacci.const", &fs::read(air_file(AirFile::Const)).unwrap());
    check_fixture("fibonacci/Fibonacci.witness.bin", witness.instances[0].stage1.trace_bytes());
    check_fixture("fibonacci/Fibonacci.qverifier.bin", &fs::read(&q_verifier_path).unwrap());
    check_fixture("fibonacci/Fibonacci.oracle.json", &json_text(&oracle_json));
}

// ---------------------------------------------------------------------------------------------
// sum_bus/
// ---------------------------------------------------------------------------------------------

/// The challenges of stage 2 of the sum bus's fixture, `[std_alpha, std_gamma]`: fixed, so that the
/// fixture is deterministic, as good as random.
fn sum_bus_challenges() -> Vec<Fr> {
    vec![Fr::from_u64(0xa1fa).pow_u64(55), Fr::from_u64(0x9a33a).pow_u64(66)]
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_sum_bus_fixtures_are_the_setups_and_the_oracles() {
    let pilout = compile_bn254("pilfflonk/tests/fixtures/sum_bus/sum_bus.pil");
    let dir = Scratch::new("sum_bus");
    let opts = SetupPilfflonkOptions {
        airout_path: dir.0.join("sum_bus.pilout"),
        build_dir: dir.0.join("build"),
        powers_of_tau: dir.0.join("tau_one.ptau"),
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: true,
    };
    fs::write(&opts.airout_path, pilout.encode_to_vec()).unwrap();
    write_tau_one_ptau(&opts.powers_of_tau, 1024).unwrap();
    run_setup_pilfflonk(&opts).unwrap();
    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    let gi = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap();
    let air_file = |file| gi.air_file(&proving_key, 0, 0, file).unwrap();
    let info = PilfflonkInfo::read(&air_file(AirFile::PilfflonkInfo)).unwrap();
    assert_eq!(info.n_stages, 2);
    let bin = fs::read(air_file(AirFile::Bin)).unwrap();
    let names: Vec<String> = Bytecode::from_bytes(&bin).unwrap().hints.into_iter().map(|h| h.name).collect();
    assert_eq!(names, ["im_col", "gsum_col"], "the std's default degree adds an im_col (plan M31)");

    // The oracle's stage 2, with the fixture's challenges.
    let witness = sum_bus::witness();
    let witness_dir = dir.0.join("witness");
    fs::create_dir_all(&witness_dir).unwrap();
    let shape = oracle::witness_shape(&pilout).unwrap();
    witness.write(&witness_dir, &shape).unwrap();
    let air_oracle = AirOracle::new(&pilout, 0, 0).unwrap();
    let source = FileWitnessSource::open(&witness_dir, &shape).unwrap();
    let mut values = air_oracle.values(&source, 0).unwrap();
    let challenges = sum_bus_challenges();
    values.challenges[1] = challenges.clone();
    air_oracle.fill_hint_columns(&mut values, 2).unwrap();
    assert!(air_oracle.check(&values).unwrap().is_empty(), "the generator's witness satisfies the AIR");
    let mut stage_2 = vec![Value::Null; info.map_sections_n["cm2"] as usize];
    for p in info.cm_pols_map.iter().filter(|p| p.stage == 2) {
        let column = if p.im_pol {
            air_oracle.expression_rows(&values, p.exp_id.unwrap() as usize).unwrap()
        } else {
            values.witness[1][p.stage_id as usize].clone()
        };
        stage_2[p.stage_pos as usize] = decs(&column);
    }
    assert!(stage_2.iter().all(|c| !c.is_null()));
    let publics: Vec<Fr> = witness.publics.iter().map(from_bytes).collect();
    let oracle_json = json!({
        "challenges": decs(&challenges),
        "publics": decs(&publics),
        "stage2": stage_2,
    });
    check_fixture("sum_bus/SumBus.bin", &bin);
    check_fixture("sum_bus/SumBus.pilfflonkinfo.json", &fs::read(air_file(AirFile::PilfflonkInfo)).unwrap());
    check_fixture("sum_bus/SumBus.const", &fs::read(air_file(AirFile::Const)).unwrap());
    check_fixture("sum_bus/SumBus.witness.bin", witness.instances[0].stage1.trace_bytes());
    let broken = sum_bus::witness_looking_up_what_is_not_provided();
    check_fixture("sum_bus/SumBus.broken.bin", broken.instances[0].stage1.trace_bytes());
    check_fixture("sum_bus/SumBus.oracle.json", &json_text(&oracle_json));
}
