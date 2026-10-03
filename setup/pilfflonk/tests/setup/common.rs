//! What the tests share: pilouts built in code (pilouts are not versioned), a
//! directory of their own, the points a ptau with `τ = 1` commits to, and the lock of the tests
//! that call the C++ core.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, constraint, expression, operand, SymbolType};
use pilfflonk_setup::ExternalFixedColumn;
use proofman_pilfflonk::{FqBytes, FrBytes, G1Affine, Layout, BN254_R};

pub const R_MINUS_ONE: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495616";
/// 2^200 + 7: a value of more than 64 bits.
pub const WIDE: &str = "1606938044258990275541962092341162602522202993782792835301383";

pub fn big(decimal: &str) -> BigUint {
    BigUint::parse_bytes(decimal.as_bytes(), 10).expect("a decimal number")
}

pub fn r() -> BigUint {
    big(BN254_R)
}

/// A value as a pilout has it: big-endian, without leading zeros (none for 0).
pub fn be(value: &BigUint) -> Vec<u8> {
    if *value == BigUint::ZERO {
        Vec::new()
    } else {
        value.to_bytes_be()
    }
}

pub fn fr(decimal: &str) -> FrBytes {
    FrBytes::from_decimal(decimal).unwrap()
}

// ---------------------------------------------------------------------------------------------
// A valid pilout: one AIR of 8 rows, 4 fixed columns (`L1`, the array `C[2]` and `U`), 2 witness
// columns of stage 1, 2 publics, a constraint that uses a constant of 254 bits and one on `C`.
// `U` is in no constraint: it is never opened.
// ---------------------------------------------------------------------------------------------

pub const N_BITS: u64 = 3;
pub const N: usize = 1 << N_BITS;

/// The fixed columns of [`pilout`]. The first rows are 1, 2, 3 and 0, so that with `τ = 1` the
/// unpacked f of each commits to G, 2G, 3G and the point at infinity (see [`multiple_of_g`]). The
/// other rows hold values of every width, 0 and `r − 1` included.
pub fn fixed_values() -> Vec<Vec<BigUint>> {
    let row = |first: u64, seed: u64| -> Vec<BigUint> {
        let mut column = vec![BigUint::from(first)];
        column.push(big(R_MINUS_ONE));
        column.push(big(WIDE) + seed);
        column.push(BigUint::ZERO);
        column.extend((4..N as u64).map(|i| BigUint::from(seed * 1000 + i) << (8 * i)));
        column
    };
    vec![row(1, 1), row(2, 2), row(3, 3), row(0, 4)]
}

pub fn fixed_column(values: &[BigUint]) -> pb::FixedCol {
    pb::FixedCol { values: values.iter().map(be).collect() }
}

pub fn op(operand: operand::Operand) -> Option<pb::Operand> {
    Some(pb::Operand { operand: Some(operand) })
}

pub fn witness(col_idx: u32, row_offset: i32) -> Option<pb::Operand> {
    op(operand::Operand::WitnessCol(operand::WitnessCol { stage: 1, col_idx, row_offset }))
}

pub fn fixed(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::FixedCol(operand::FixedCol { idx, row_offset: 0 }))
}

pub fn constant(value: &BigUint) -> Option<pb::Operand> {
    op(operand::Operand::Constant(operand::Constant { value: be(value) }))
}

pub fn exp(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::Expression(operand::Expression { idx }))
}

pub fn sub(lhs: Option<pb::Operand>, rhs: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Sub(expression::Sub { lhs, rhs })) }
}

pub fn mul(lhs: Option<pb::Operand>, rhs: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Mul(expression::Mul { lhs, rhs })) }
}

pub fn neg(value: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Neg(expression::Neg { value })) }
}

pub fn every_row(idx: u32) -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::EveryRow(constraint::EveryRow {
            expression_idx: Some(operand::Expression { idx }),
            debug_line: Some(format!("constraint on expression {idx}")),
        })),
    }
}

pub fn symbol(name: &str, r#type: SymbolType, id: u32, stage: Option<u32>, air: bool) -> pb::Symbol {
    pb::Symbol {
        name: name.to_string(),
        air_group_id: air.then_some(0),
        air_id: air.then_some(0),
        r#type: r#type as i32,
        id,
        stage,
        dim: 1,
        lengths: vec![],
        commit_id: None,
        debug_line: None,
    }
}

/// The symbol of an array column of `length` elements, the first at `id`.
pub fn array_symbol(name: &str, r#type: SymbolType, id: u32, stage: u32, length: u32) -> pb::Symbol {
    pb::Symbol { lengths: vec![length], ..symbol(name, r#type, id, Some(stage), true) }
}

/// A hint of the AIR, or of the pilout when `air` is false.
pub fn hint(name: &str, air: bool) -> pb::Hint {
    pb::Hint { name: name.to_string(), hint_fields: vec![], air_group_id: air.then_some(0), air_id: air.then_some(0) }
}

/// The valid pilout of the tests. Its first constraint, `−((a − L1)·(r − 1)·b')`, has a constant
/// of 254 bits; the second is `C[0] − C[1]`.
pub fn pilout() -> pb::PilOut {
    let air = pb::Air {
        name: Some("Sample".into()),
        num_rows: Some(N as u32),
        fixed_cols: fixed_values().iter().map(|c| fixed_column(c)).collect(),
        stage_widths: vec![2],
        expressions: vec![
            sub(witness(0, 0), fixed(0)),
            mul(exp(0), constant(&big(R_MINUS_ONE))),
            mul(exp(1), witness(1, 1)),
            neg(exp(2)),
            sub(fixed(1), fixed(2)),
        ],
        constraints: vec![every_row(3), every_row(4)],
        ..Default::default()
    };
    pb::PilOut {
        name: Some("Synthetic".into()),
        base_field: r().to_bytes_be(),
        air_groups: vec![pb::AirGroup { name: Some("Group".into()), air_group_values: vec![], airs: vec![air] }],
        num_challenges: vec![0],
        num_proof_values: vec![0],
        num_public_values: 2,
        symbols: vec![
            symbol("in", SymbolType::PublicValue, 0, None, false),
            symbol("out", SymbolType::PublicValue, 1, None, false),
            symbol("Sample.L1", SymbolType::FixedCol, 0, Some(0), true),
            array_symbol("Sample.C", SymbolType::FixedCol, 1, 0, 2),
            symbol("Sample.U", SymbolType::FixedCol, 3, Some(0), true),
            symbol("Sample.a", SymbolType::WitnessCol, 0, Some(1), true),
            symbol("Sample.b", SymbolType::WitnessCol, 1, Some(1), true),
        ],
        ..Default::default()
    }
}

pub fn the_air(pilout: &mut pb::PilOut) -> &mut pb::Air {
    &mut pilout.air_groups[0].airs[0]
}

/// [`pilout`] without the values of `C[0]`, `C[1]` and `U`, as pil2com writes the columns it
/// declares `#pragma fixed_external` (pilfflonk/docs/formats.md#fixed-columns), and those values as
/// external columns, in the order of the columns.
pub fn pilout_with_external_fixed() -> (pb::PilOut, Vec<ExternalFixedColumn>) {
    let mut pilout = pilout();
    for column in 1..4 {
        the_air(&mut pilout).fixed_cols[column].values.clear();
    }
    let values = fixed_values();
    let external = |name: &str, index: usize, column: usize| ExternalFixedColumn {
        name: name.to_string(),
        index,
        values: values[column].iter().map(|v| fr(&v.to_str_radix(10))).collect(),
    };
    (pilout, vec![external("Sample.C", 0, 1), external("Sample.C", 1, 2), external("Sample.U", 0, 3)])
}

/// An `f` of a layout as `(stage, the names of its polynomials, k, offsets, degree)`.
pub type FShape<'a> = (u64, Vec<&'a str>, u64, Vec<i64>, u64);

/// Each `f` of `layout` as an [`FShape`].
pub fn f_shapes(layout: &Layout) -> Vec<FShape<'_>> {
    layout
        .0
        .iter()
        .map(|f| (f.stage, f.pols.iter().map(|p| p.name.as_str()).collect(), f.k, f.offsets.clone(), f.degree))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Directories and points
// ---------------------------------------------------------------------------------------------

/// Held for the whole of every test that calls the C++ core's OpenMP code (the SRS, the fixed
/// commitments, `run_setup_pilfflonk`), so that those tests run one at a time
/// (pilfflonk/docs/README.md#tests).
///
/// Each test runs on a thread of its own, which OpenMP makes a root with its own team, a thread per
/// CPU, kept until that thread exits. Enough such tests at once outgrow libomp's table of threads
/// (4 per CPU at first): libomp 14, Ubuntu 22.04's, then allocates a larger table and frees the old
/// one while the workers it has just started may still be reading it (their stack-overlap check),
/// and the process dies of SIGSEGV now and then (4 runs of this binary in 262 on 256 CPUs, the
/// worker scanning a freed `__kmp_threads`). With this lock, the teams alive are at most those of the
/// test that holds it and of the one that has just let it go, and the table never grows.
pub fn cpp_core() -> MutexGuard<'static, ()> {
    static CPP_CORE: Mutex<()> = Mutex::new(());
    // A test that failed while holding it poisoned it; the next ones run all the same.
    CPP_CORE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A fresh directory for one test under the target's temporary directory, removed when it is
/// dropped (a failed test leaves it behind, under target/).
pub struct TestDir(pub PathBuf);

impl TestDir {
    pub fn new(name: &str) -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("pilfflonk_setup_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        TestDir(dir)
    }

    pub fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// An affine point from big-endian hex coordinates.
fn point(x: &str, y: &str) -> G1Affine {
    let coordinate = |hex: &str| {
        let bytes = BigUint::parse_bytes(hex.as_bytes(), 16).unwrap().to_bytes_be();
        let mut be = [0u8; 32];
        be[32 - bytes.len()..].copy_from_slice(&bytes);
        FqBytes::from_be_bytes(be).unwrap()
    };
    G1Affine { x: coordinate(x), y: coordinate(y) }
}

/// `m·G` for the generator `G = (1, 2)` of G1, for the `m` the tests need: BN254's well-known
/// multiples, written out independently of the C++ core (as `provers/starks-lib-c/tests/
/// pilfflonk_srs.rs` has them); `0·G` is the point at infinity, `(0, 0)`.
pub fn multiple_of_g(m: u64) -> G1Affine {
    match m {
        0 => G1Affine::INFINITY,
        1 => point("1", "2"),
        2 => point(
            "030644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd3",
            "15ed738c0e0a7c92e7845f96b2ae9c0a68a6a449e3538fc7ff3ebf7a5a18a2c4",
        ),
        3 => point(
            "0769bf9ac56bea3ff40232bcb1b6bd159315d84715b8e679f2d355961915abf0",
            "2ab799bee0489429554fdb7c8d086475319e63b40b9c5b57cdf1ff3dd9fe2261",
        ),
        5 => point(
            "17c139df0efee0f766bc0204762b774362e4ded88953a39ce849a8a7fa163fa9",
            "01e0559bacb160664764a357af8a9fe70baa9258e0b959273ffc5718c6d4cc7c",
        ),
        _ => panic!("{m}·G is not written out here"),
    }
}

/// The generator of G2, canonical, `x.c0, x.c1, y.c0, y.c1` in decimal: the `[1]₂` of every
/// BN254 library, and `[τ]₂` of a ptau with `τ = 1`.
pub const G2_GENERATOR: [&str; 4] = [
    "10857046999023057135944570762232829481370756359578518086990519993285655852781",
    "11559732032986387107991004021392285783925812861821192530917403151452391805634",
    "8495653923123431417604973247489272438418190587263600148770280649306958101930",
    "4082367875863433681332203403145435568316851327593401208105741076214120093531",
];
