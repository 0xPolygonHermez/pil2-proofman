//! The range checks of the BN128 verifier's custom mode, `circuits.bn128/custom/rangecheck.circom`
//! and `custom/lessthangl.circom`, on `fixtures/bn128/range_check.circom`: a `RangeCheck` of each
//! width that matters to `Num2Bytes`, and a `LessThanGoldilocks`.
//!
//! `Num2Bytes` is a custom gate, which has no r1cs constraint: circom computes its 16-bit chunks and
//! its witness refuses a value out of range, and plonk2pil's gate is what constrains them. So the
//! tests hold the witness to the bounds, exactly, and the r1cs to what the gate takes: per use, a
//! `Num2Bytes(nBits)` with `nBits` at most 80 and the signals `[in, out[0], …, out[nB − 1]]`,
//! `nB = ceil(nBits/16)`, `in` the value checked and `out` its chunks; above 80 bits, two uses, on
//! the low 80 bits and on the rest, which an r1cs constraint adds back up.
//!
//! They need what the witness needs (`common`); without it they say why and pass.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use num_bigint::BigUint;
use pil2_stark_recurser::plonk2pil::r1cs::types::{read_r1cs_from_bytes, LinearCombination, R1csConstraint, R1csFile};
use proofman_fields::{Bn128, Field, PrimeField};

use common::{circom, circuits_bn128, manifest, missing_prerequisite, read_wtns, run, snarkjs};

/// The widths of the circuit's `RangeCheck`s, input by input.
const WIDTHS: [u32; 9] = [1, 16, 17, 64, 65, 80, 81, 154, 160];

/// The Goldilocks prime.
const P: u64 = 0xFFFF_FFFF_0000_0001;

/// `2^n − 1`.
fn ones(n: u32) -> BigUint {
    (BigUint::from(1u32) << n) - 1u32
}

/// The circuit's inputs: the `RangeCheck`s', then the `LessThanGoldilocks`'s.
struct Inputs {
    range: Vec<BigUint>,
    gl: BigUint,
}

impl Inputs {
    fn zero() -> Self {
        Self { range: vec![BigUint::ZERO; WIDTHS.len()], gl: BigUint::ZERO }
    }

    /// The largest values in range.
    fn max() -> Self {
        Self { range: WIDTHS.iter().map(|&n| ones(n)).collect(), gl: BigUint::from(P - 1) }
    }

    fn json(&self) -> String {
        let decimal = |v: &BigUint| format!("\"{v}\"");
        let range: Vec<String> = self.range.iter().map(decimal).collect();
        format!("{{\"in\": [{}], \"gl\": {}}}", range.join(", "), decimal(&self.gl))
    }
}

/// The circuit, compiled once for the tests of this binary.
struct Circuit {
    dir: PathBuf,
    r1cs: R1csFile<Bn128>,
}

fn circuit() -> Option<&'static Circuit> {
    static CIRCUIT: OnceLock<Option<Circuit>> = OnceLock::new();
    CIRCUIT
        .get_or_init(|| match missing_prerequisite() {
            Some(why) => {
                eprintln!("skipping the range check tests: {why}");
                None
            }
            None => Some(Circuit::compile()),
        })
        .as_ref()
}

impl Circuit {
    /// As `common::compile_and_witness` compiles a circuit.
    fn compile() -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("range_check_bn128");
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        run(
            Command::new(circom())
                .args(["--O1", "--r1cs", "--wasm", "--prime", "bn128", "-l"])
                .arg(circuits_bn128())
                .arg(manifest().join("tests/fixtures/bn128/range_check.circom"))
                .arg("-o")
                .arg(&dir),
            "circom",
        );
        let r1cs = read_r1cs_from_bytes(&fs::read(dir.join("range_check.r1cs")).unwrap()).unwrap();
        Self { dir, r1cs }
    }

    /// The witness of `inputs`, or what snarkjs says when the circuit refuses them. `tag` names its
    /// files.
    fn witness(&self, tag: &str, inputs: &Inputs) -> Result<Vec<Bn128>, String> {
        let (input, wtns) = (self.dir.join(format!("{tag}.json")), self.dir.join(format!("{tag}.wtns")));
        fs::write(&input, inputs.json()).unwrap();
        let out = Command::new("node")
            .arg(snarkjs())
            .args(["wtns", "calculate"])
            .arg(self.dir.join("range_check_js/range_check.wasm"))
            .arg(&input)
            .arg(&wtns)
            .output()
            .expect("run snarkjs");
        if !out.status.success() {
            return Err(format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)));
        }
        Ok(read_wtns(&fs::read(&wtns).unwrap()))
    }

    /// The r1cs constraints `witness` breaks.
    fn failing_constraints(&self, witness: &[Bn128]) -> Vec<usize> {
        let eval = |lc: &LinearCombination<Bn128>| {
            lc.iter().fold(Bn128::ZERO, |acc, (&wire, &q)| acc + q * witness[wire as usize])
        };
        let holds = |c: &R1csConstraint<Bn128>| eval(&c.a) * eval(&c.b) == eval(&c.c);
        self.r1cs.constraints.iter().enumerate().filter(|(_, c)| !holds(c)).map(|(i, _)| i).collect()
    }

    /// Every use of the witness, as `(nBits, in)`, once it is checked to be a `Num2Bytes(nBits)` of
    /// signals `[in, out…]`: `nBits` at most 80, and `out` the `ceil(nBits/16)` chunks of `in`, each
    /// below 2^16, which add up to it.
    fn uses(&self, witness: &[Bn128]) -> Vec<(u32, BigUint)> {
        let value = |wire: u64| witness[wire as usize].as_canonical_biguint();
        let mut uses: Vec<(u32, BigUint)> = self
            .r1cs
            .custom_gates_uses
            .iter()
            .map(|gate_use| {
                let gate = &self.r1cs.custom_gates[gate_use.id as usize];
                assert_eq!(gate.template_name, "Num2Bytes");
                let [n_bits] = gate.parameters[..] else { panic!("Num2Bytes takes nBits: {:?}", gate.parameters) };
                let n_bits = u32::try_from(&n_bits.as_canonical_biguint()).unwrap();
                assert!(n_bits <= 80, "Num2Bytes({n_bits})");

                let (&input, chunks) = gate_use.signals.split_first().expect("the signals of a use");
                assert_eq!(chunks.len(), n_bits.div_ceil(16) as usize, "the chunks of Num2Bytes({n_bits})");
                let mut sum = BigUint::ZERO;
                for (k, &chunk) in chunks.iter().enumerate() {
                    assert!(value(chunk) < BigUint::from(1u32 << 16), "chunk {k} of Num2Bytes({n_bits})");
                    sum += value(chunk) << (16 * k);
                }
                assert_eq!(sum, value(input), "the chunks of Num2Bytes({n_bits}) add up to its in");
                (n_bits, value(input))
            })
            .collect();
        uses.sort();
        uses
    }
}

#[test]
fn range_checks_accept_the_ends_of_their_ranges() {
    let Some(circuit) = circuit() else { return };
    for (tag, inputs) in [("zero", Inputs::zero()), ("max", Inputs::max())] {
        let witness = circuit.witness(tag, &inputs).unwrap_or_else(|e| panic!("{tag}: {e}"));
        assert_eq!(circuit.failing_constraints(&witness), Vec::<usize>::new(), "{tag}");
        // The circuit's output, LessThanGoldilocks's.
        assert_eq!(witness[1].as_canonical_biguint(), inputs.gl, "{tag}");
    }
}

#[test]
fn range_checks_refuse_their_bound_in_the_witness() {
    let Some(circuit) = circuit() else { return };
    for (i, &n) in WIDTHS.iter().enumerate() {
        let mut inputs = Inputs::max();
        inputs.range[i] = BigUint::from(1u32) << n;
        let err = circuit.witness(&format!("bound_{n}"), &inputs).expect_err(&format!("RangeCheck({n}) takes 2^{n}"));
        assert!(err.contains("Assert Failed") && err.contains("Num2Bytes"), "RangeCheck({n}): {err}");
        assert!(err.contains("RangeCheck"), "RangeCheck({n}): {err}");
    }
}

#[test]
fn less_than_goldilocks_accepts_p_minus_1_and_refuses_p() {
    let Some(circuit) = circuit() else { return };
    let mut inputs = Inputs::zero();
    inputs.gl = BigUint::from(P - 1);
    circuit.witness("p_minus_1", &inputs).unwrap();
    inputs.gl = BigUint::from(P);
    let err = circuit.witness("p", &inputs).expect_err("LessThanGoldilocks takes p");
    assert!(err.contains("Assert Failed") && err.contains("LessThanGoldilocks"), "{err}");
}

#[test]
fn every_use_is_a_num2bytes_of_a_value_and_its_16_bit_chunks() {
    let Some(circuit) = circuit() else { return };
    let mut witness = circuit.witness("uses", &Inputs::max()).unwrap();

    // One use per RangeCheck up to 80 bits, and two above, on the low 80 bits and the rest; two for
    // LessThanGoldilocks, on its in and on in + 2^64 − p.
    let mut expected: Vec<(u32, BigUint)> = WIDTHS
        .iter()
        .flat_map(|&n| if n <= 80 { vec![(n, ones(n))] } else { vec![(80, ones(80)), (n - 80, ones(n - 80))] })
        .collect();
    expected.extend([(64, BigUint::from(P - 1)), (64, ones(64))]);
    expected.sort();
    assert_eq!(circuit.uses(&witness), expected);

    // The r1cs adds the two parts of a wide RangeCheck back up to its value: the high part of
    // RangeCheck(154), its only Num2Bytes(74), changed, breaks it.
    let (gates, n_bits) = (&circuit.r1cs.custom_gates, [Bn128::from_decimal("74").unwrap()]);
    let high = circuit
        .r1cs
        .custom_gates_uses
        .iter()
        .find(|gate_use| gates[gate_use.id as usize].parameters == n_bits)
        .expect("the Num2Bytes(74) of RangeCheck(154)");
    witness[high.signals[0] as usize] += Bn128::ONE;
    assert_eq!(circuit.failing_constraints(&witness).len(), 1);
}
