//! The Fibonacci fixture (pilfflonk/docs/README.md#fixtures): the witness generator of
//! `tests/data/fibonacci.rs` against pil-fflonk's publics, its witness directory, and the Rust
//! oracle (pilfflonk/docs/README.md#tests) on the pilout of
//! `tests/fixtures/fibonacci/fibonacci.pil`.
//!
//! Pilouts are not versioned: the `#[ignore]` tests compile the fixture with the compiler
//! `PIL2C_EXEC` names, which must have `--field` (the pinned one silently compiles over
//! Goldilocks):
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p proofman-pilfflonk --features proofman-common/cpu-only \
//!     --test fibonacci -- --include-ignored
//! ```

mod data {
    pub mod fibonacci;
}

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use num_bigint::BigUint;
use pil2_pilout::pilout as pb;
use pil2_pilout::pilout_proxy::PilOutProxy;
use proofman_pilfflonk::oracle::{self, AirOracle, Domain, Fr, Values};
use proofman_pilfflonk::{
    AirInstanceRef, AirShape, FileWitnessSource, FrBytes, Witness, WitnessShape, WitnessSource, BN128_R,
};

use data::fibonacci;

/// The fixture's size and inputs, those of pil-fflonk's `all` example
/// (pilfflonk/docs/README.md#fixtures).
const N_BITS: u32 = 8;
const INPUTS: [u64; 2] = [1, 2];

/// `pil-fflonk/runtime/public.json`, the publics pil-fflonk's prover gave for this fixture.
const PIL_FFLONK_PUBLICS: [&str; 3] =
    ["1", "2", "590308608561184158373097535019708483037277117989374906445627411437315467687"];

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("fibonacci").join(name);
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    dir
}

fn big(v: &FrBytes) -> BigUint {
    BigUint::from_bytes_le(&v.to_le_bytes())
}

/// The shape of the fixture, written out: one AIR of 2^8 rows and the stage-1 columns `l1`, `l2`,
/// no air values, the publics `in1`, `in2`, `out` and no proof values.
fn fibonacci_shape() -> WitnessShape {
    let air = AirShape { airgroup_id: 0, air_id: 0, n_bits: N_BITS as u64, n_cols: 2, n_air_values: 0 };
    WitnessShape::new(vec![air], 3, 0).unwrap()
}

#[test]
fn the_generator_gives_pil_fflonks_publics() {
    let witness = fibonacci::witness(N_BITS, INPUTS);
    let publics: Vec<String> = witness.publics.iter().map(FrBytes::to_decimal).collect();
    assert_eq!(publics, PIL_FFLONK_PUBLICS);
    assert!(witness.proof_values.is_empty());
}

#[test]
fn the_generator_follows_sm_fibonacci_js() {
    let witness = fibonacci::witness(N_BITS, INPUTS);
    assert_eq!(witness.instances.len(), 1);
    let instance = &witness.instances[0];
    assert_eq!(instance.air, AirInstanceRef { airgroup_id: 0, air_id: 0 });
    let stage1 = &instance.stage1;
    assert_eq!((stage1.n_rows(), stage1.n_cols()), (256, 2));
    assert!(stage1.air_values().is_empty());
    let l1: Vec<BigUint> = stage1.column(0).unwrap().iter().map(big).collect();
    let l2: Vec<BigUint> = stage1.column(1).unwrap().iter().map(big).collect();

    // The first rows by hand: l2 = 1, 2, 5, 29; l1 = 2, 1 + 4, 4 + 25, 25 + 841.
    let small = |v: &[BigUint]| v[..4].iter().map(|x| x.to_string()).collect::<Vec<_>>();
    assert_eq!(small(&l2), ["1", "2", "5", "29"]);
    assert_eq!(small(&l1), ["2", "5", "29", "866"]);

    // Every row, recomputed here: l2[i] = l1[i-1] and l1[i] = l2[i-1]² + l1[i-1]² mod r.
    let r = BigUint::parse_bytes(BN128_R.as_bytes(), 10).unwrap();
    for i in 1..256 {
        assert_eq!(l2[i], l1[i - 1], "row {i}");
        assert_eq!(l1[i], (l2[i - 1].pow(2) + l1[i - 1].pow(2)) % &r, "row {i}");
    }
    assert!(l1[255] > BigUint::from(u64::MAX), "the values wrap around r: they are not small");
    assert_eq!(l1[255].to_string(), PIL_FFLONK_PUBLICS[2], "out = l1(N-1)");
    assert_eq!((l2[0].to_string(), l1[0].to_string()), ("1".into(), "2".into()), "in1 = l2(0), in2 = l1(0)");
}

#[test]
fn the_fibonacci_witness_round_trips_through_its_directory() {
    let dir = scratch("witness");
    let witness = fibonacci::witness(N_BITS, INPUTS);
    witness.write(&dir, &fibonacci_shape()).unwrap();
    assert_eq!(fs::metadata(dir.join("instance_0_0_0.bin")).unwrap().len(), 256 * 2 * 32);
    let source = FileWitnessSource::open(&dir, &fibonacci_shape()).unwrap();
    assert_eq!(Witness::from_source(&source).unwrap(), witness);
    let publics: Vec<String> = source.publics().unwrap().iter().map(FrBytes::to_decimal).collect();
    assert_eq!(publics, PIL_FFLONK_PUBLICS);
}

// ---------------------------------------------------------------------------------------------
// The oracle on the compiled fixture (needs PIL2C_EXEC)
// ---------------------------------------------------------------------------------------------

const N: usize = 1 << N_BITS;

/// The expression of `l1' − next`, `next = l1² + l2²`: the im pol the setup chooses at its default
/// degree (`setup/pil-info/tests/bn128.rs`, `fibonacci_fixture_over_bn128`), which leaves
/// `qDeg = 1` (pilfflonk/docs/protocol.md#degree-search).
const L1_NEXT_MINUS_NEXT: usize = 6;

/// The fixture compiled over BN128, once for all the tests of this binary.
fn pilout() -> &'static pb::PilOut {
    static PILOUT: OnceLock<pb::PilOut> = OnceLock::new();
    PILOUT.get_or_init(compile_fibonacci)
}

/// Compiles the fixture over BN128 with `PIL2C_EXEC`, as `setup/pil-info/tests/bn128.rs` does.
fn compile_fibonacci() -> pb::PilOut {
    let compiler = std::env::var("PIL2C_EXEC")
        .expect("PIL2C_EXEC must name a pil2com that has `--field` (e.g. <pil2-compiler>/src/pil.js)");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().expect("the repository root");
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("fibonacci.bn128.pilout");
    let status = Command::new(compiler)
        .current_dir(&root)
        .arg("pilfflonk/tests/fixtures/fibonacci/fibonacci.pil")
        .arg("-I")
        .arg("pil2-components/lib/std/pil")
        .arg("--field")
        .arg("bn128")
        .arg("-o")
        .arg(&out)
        .status()
        .expect("PIL2C_EXEC runs");
    assert!(status.success(), "pil2com failed on the fixture");
    PilOutProxy::new(out.to_str().expect("a UTF-8 path")).expect("a pilout").pilout
}

/// The oracle of the fixture, and the values of the generator's witness read back from its
/// directory by `FileWitnessSource`, against the shape of the pilout.
fn oracle_and_values() -> (AirOracle, Values) {
    static CELL: OnceLock<(AirOracle, Values)> = OnceLock::new();
    let (oracle, values) = CELL.get_or_init(|| {
        let oracle = AirOracle::new(pilout(), 0, 0).expect("a pilout over BN128");
        let dir = scratch("oracle_witness");
        fibonacci::witness(N_BITS, INPUTS).write(&dir, &fibonacci_shape()).unwrap();
        let source = FileWitnessSource::open(&dir, &oracle::witness_shape(pilout()).unwrap()).unwrap();
        let values = oracle.values(&source, 0).unwrap();
        (oracle, values)
    });
    (oracle.clone(), values.clone())
}

/// A cell to mutate, `(column, row)`, and the `(constraint, row)` that must then fail.
type Mutation = (usize, usize, &'static [(usize, usize)]);

fn failures(oracle: &AirOracle, values: &Values) -> BTreeSet<(usize, usize)> {
    oracle.check(values).unwrap().into_iter().map(|f| (f.constraint, f.row)).collect()
}

/// Points off `H`, fixed so that the tests are deterministic: as good as random ones.
fn points() -> [Fr; 2] {
    [Fr::from_u64(7).pow_u64(1000), -&Fr::from_u64(123456789)]
}

fn std_vc() -> Fr {
    Fr::from_u64(0x5eed).pow_u64(77)
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_pilout_has_the_shape_and_fixed_columns_of_the_fixture() {
    let pilout = pilout();
    assert_eq!(oracle::witness_shape(pilout).unwrap(), fibonacci_shape());
    let oracle = AirOracle::new(pilout, 0, 0).unwrap();
    assert_eq!(oracle.n_rows(), N);

    // sm_fibonacci.js's buildConstants: L1 = [1, 0, …], LLAST = [0, …, 0, 1].
    let unit = |row: usize| (0..N).map(|i| Fr::from_u64(u64::from(i == row))).collect::<Vec<_>>();
    assert_eq!(oracle.fixed(0).unwrap(), unit(0), "L1");
    assert_eq!(oracle.fixed(1).unwrap(), unit(N - 1), "LLAST");
    assert!(oracle.fixed(2).is_none());

    let constraints: Vec<(Domain, &str)> =
        oracle.constraints().iter().map(|c| (c.domain, c.debug_line.as_str())).collect();
    assert_eq!(
        constraints,
        [
            (Domain::EveryRow, "fibonacci.pil:24 (l2'-l1)*(1-Fibonacci.LLAST)"),
            (Domain::EveryRow, "fibonacci.pil:27 (l1'-((l1*l1)+(l2*l2)))*(1-Fibonacci.LLAST)"),
            (Domain::EveryRow, "fibonacci.pil:29 Fibonacci.L1*(l2-in1)"),
            (Domain::EveryRow, "fibonacci.pil:30 Fibonacci.L1*(l1-in2)"),
            (Domain::EveryRow, "fibonacci.pil:31 Fibonacci.LLAST*(l1-out)"),
        ]
    );
    assert_eq!(oracle.constraints()[1].expression, L1_NEXT_MINUS_NEXT + 2, "(l1' − next)·(1 − LLAST)");
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn the_generators_witness_makes_every_numerator_0_on_every_row() {
    let (oracle, values) = oracle_and_values();
    let numerators = oracle.numerators(&values).unwrap();
    assert_eq!(numerators.len(), 5);
    for (c, rows) in numerators.iter().enumerate() {
        assert_eq!(rows.len(), N);
        assert!(rows.iter().all(Fr::is_zero), "constraint {c} is not 0 on some row");
    }
    assert_eq!(oracle.check(&values).unwrap(), []);
    // The publics are in1 = l2(0), in2 = l1(0) and out = l1(N − 1).
    assert_eq!(values.publics[0], values.witness[0][1][0]);
    assert_eq!(values.publics[1], values.witness[0][0][0]);
    assert_eq!(values.publics[2], values.witness[0][0][N - 1]);
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn a_mutated_cell_fails_exactly_the_constraints_and_rows_that_read_it() {
    let (oracle, good) = oracle_and_values();
    const L1: usize = 0;
    const L2: usize = 1;
    // Constraints: 0 (l2' − l1)(1 − LLAST), 1 (l1' − l1² − l2²)(1 − LLAST), 2 L1(l2 − in1),
    // 3 L1(l1 − in2), 4 LLAST(l1 − out). Row N − 1 reads row 0 through the offset, and 1 − LLAST
    // masks it.
    let cases: [Mutation; 7] = [
        (L1, 100, &[(0, 100), (1, 99), (1, 100)]),
        (L2, 100, &[(0, 99), (1, 100)]),
        (L1, 0, &[(0, 0), (1, 0), (3, 0)]),
        (L2, 0, &[(1, 0), (2, 0)]),
        (L1, N - 1, &[(1, N - 2), (4, N - 1)]),
        (L2, N - 1, &[(0, N - 2)]),
        (L1, 1, &[(0, 1), (1, 0), (1, 1)]),
    ];
    for (col, row, expected) in cases {
        let mut values = good.clone();
        values.witness[0][col][row] = &values.witness[0][col][row] + &Fr::one();
        let found = failures(&oracle, &values);
        let described: Vec<String> = found
            .iter()
            .map(|(c, r)| format!("constraint {c} ({}) at row {r}", oracle.constraints()[*c].debug_line))
            .collect();
        println!("{} [{row}] + 1: {described:?}", ["l1", "l2"][col]);
        assert_eq!(found, expected.iter().copied().collect(), "column {col}, row {row}");
    }

    let mut values = good.clone();
    values.publics[2] = &values.publics[2] + &Fr::one();
    assert_eq!(failures(&oracle, &values), [(4, N - 1)].into_iter().collect(), "out");
    let mut values = good;
    values.publics[0] = &values.publics[0] + &Fr::one();
    assert_eq!(failures(&oracle, &values), [(2, 0)].into_iter().collect(), "in1");
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn q_from_barycentric_evaluations_is_the_exact_quotient() {
    let (oracle, values) = oracle_and_values();
    let std_vc = std_vc();

    // Without im pols, the degree-3 constraints give deg Q ≤ 3(N − 1) − N = 2N − 3: qDeg = 2.
    let q = oracle.q_polynomial(&values, &[], &std_vc).unwrap();
    assert!(q.is_exact(), "remainders on the terms {:?}", q.remainders.iter().map(|(t, _)| t).collect::<Vec<_>>());
    assert!(q.coefficients.len() > N && q.coefficients.len() <= 2 * N - 2, "{} coefficients", q.coefficients.len());
    for z in points() {
        assert_eq!(oracle.q_at(&values, &[], &std_vc, &z).unwrap(), q.evaluate(&z));
    }

    // With the im pol of `l1' − next`, every term has degree 2: deg Q ≤ 2(N − 1) − N = N − 2,
    // qDeg = 1, as pilfflonk/docs/protocol.md#degree-search says of this fixture.
    let im = [L1_NEXT_MINUS_NEXT];
    let q_im = oracle.q_polynomial(&values, &im, &std_vc).unwrap();
    assert!(q_im.is_exact());
    assert!(!q_im.coefficients.is_empty() && q_im.coefficients.len() < N, "{} coefficients", q_im.coefficients.len());
    for z in points() {
        assert_eq!(oracle.q_at(&values, &im, &std_vc, &z).unwrap(), q_im.evaluate(&z));
    }
    assert_ne!(q_im.evaluate(&points()[0]), q.evaluate(&points()[0]), "another fold: another polynomial");
    println!(
        "Q: {} coefficients without im pols, {} with the im pol {L1_NEXT_MINUS_NEXT} (N = {N})",
        q.coefficients.len(),
        q_im.coefficients.len()
    );
}

#[test]
#[ignore = "needs PIL2C_EXEC"]
fn a_mutated_witness_gives_no_polynomial_q() {
    let (oracle, good) = oracle_and_values();
    let std_vc = std_vc();
    let mut values = good;
    values.witness[0][0][100] = &values.witness[0][0][100] + &Fr::one();
    for im in [&[][..], &[L1_NEXT_MINUS_NEXT][..]] {
        let q = oracle.q_polynomial(&values, im, &std_vc).unwrap();
        let terms: Vec<usize> = q.remainders.iter().map(|(t, _)| *t).collect();
        assert_eq!(terms, [0, 1], "im pols {im:?}: the two transition constraints do not vanish on H");
        for z in points() {
            assert_ne!(oracle.q_at(&values, im, &std_vc, &z).unwrap(), q.evaluate(&z), "im pols {im:?}");
        }
    }
}
