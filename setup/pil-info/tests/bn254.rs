//! The passes over BN254 (`PilInfoCfg::bn254()`): every value has dimension 1, a negation is a
//! multiplication by `r − 1`, constants keep all their bits, there is no FRI polynomial, and the
//! degree search follows `DegreePolicy::Search`.
//!
//! The pilouts are built in code; pilouts are not versioned. The `#[ignore]` tests compile real PIL
//! with the compiler `PIL2C_EXEC` names, which must honour `prime` (the pinned one silently
//! compiles over Goldilocks):
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p pil-info --test bn254 -- --ignored
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, constraint, expression, operand, SymbolType};
use pil2_pilout::pilout_proxy::PilOutProxy;
use pil_info::output::expressions_info::build_verifier_info_json;
use pil_info::types::output::CodeEntry;
use pil_info::{DegreePolicy, PilInfoCfg, PilInfoResult};

const R: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";
const R_MINUS_ONE: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495616";
const R_MINUS_TWO: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495615";
/// 2^200 + 7.
const WIDE: &str = "1606938044258990275541962092341162602522202993782792835301383";
/// Goldilocks p − 1: what a `neg` became before the passes were parameterised.
const GOLDILOCKS_NEG_ONE: &str = "18446744069414584320";

// ---------------------------------------------------------------------------
// Pilouts built in code
// ---------------------------------------------------------------------------

fn big_be(decimal: &str) -> Vec<u8> {
    BigUint::parse_bytes(decimal.as_bytes(), 10).expect("a decimal constant").to_bytes_be()
}

fn op(operand: operand::Operand) -> Option<pb::Operand> {
    Some(pb::Operand { operand: Some(operand) })
}

fn witness(col_idx: u32, row_offset: i32) -> Option<pb::Operand> {
    op(operand::Operand::WitnessCol(operand::WitnessCol { stage: 1, col_idx, row_offset }))
}

fn fixed(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::FixedCol(operand::FixedCol { idx, row_offset: 0 }))
}

fn constant(decimal: &str) -> Option<pb::Operand> {
    op(operand::Operand::Constant(operand::Constant { value: big_be(decimal) }))
}

fn exp(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::Expression(operand::Expression { idx }))
}

fn add(lhs: Option<pb::Operand>, rhs: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Add(expression::Add { lhs, rhs })) }
}

fn sub(lhs: Option<pb::Operand>, rhs: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Sub(expression::Sub { lhs, rhs })) }
}

fn mul(lhs: Option<pb::Operand>, rhs: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Mul(expression::Mul { lhs, rhs })) }
}

fn neg(value: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Neg(expression::Neg { value })) }
}

fn every_row(idx: u32) -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::EveryRow(constraint::EveryRow {
            expression_idx: Some(operand::Expression { idx }),
            debug_line: Some(format!("constraint on expression {idx}")),
        })),
    }
}

fn column_symbol(name: &str, kind: SymbolType, stage: u32, id: u32) -> pb::Symbol {
    pb::Symbol {
        name: name.to_string(),
        air_group_id: Some(0),
        air_id: Some(0),
        r#type: kind as i32,
        id,
        stage: Some(stage),
        ..Default::default()
    }
}

/// One BN254 air of 16 rows with one stage (`numChallenges = [0]`), as pil2com emits it.
fn pilout(
    witness_names: &[&str],
    fixed_names: &[&str],
    expressions: Vec<pb::Expression>,
    constraints: Vec<pb::Constraint>,
) -> pb::PilOut {
    let mut symbols: Vec<pb::Symbol> = fixed_names
        .iter()
        .enumerate()
        .map(|(i, name)| column_symbol(name, SymbolType::FixedCol, 0, i as u32))
        .collect();
    symbols.extend(
        witness_names.iter().enumerate().map(|(i, name)| column_symbol(name, SymbolType::WitnessCol, 1, i as u32)),
    );
    let air = pb::Air {
        name: Some("Synthetic".to_string()),
        num_rows: Some(16),
        fixed_cols: fixed_names.iter().map(|_| pb::FixedCol { values: Vec::new() }).collect(),
        stage_widths: vec![witness_names.len() as u32],
        expressions,
        constraints,
        ..Default::default()
    };
    pb::PilOut {
        name: Some("synthetic".to_string()),
        base_field: big_be(R),
        air_groups: vec![pb::AirGroup { name: Some("Synthetic".to_string()), airs: vec![air], ..Default::default() }],
        num_challenges: vec![0],
        symbols,
        ..Default::default()
    }
}

/// Wide constants and negations: `L1·(a − (2^200 + 7))`, `L1·(b − (r − 2))`, `(−a)·b + b'` and
/// `−(a'·a) + b`.
fn wide_constants_pilout() -> pb::PilOut {
    pilout(
        &["a", "b"],
        &["L1"],
        vec![
            sub(witness(0, 0), constant(WIDE)),        // 0
            mul(fixed(0), exp(0)),                     // 1
            sub(witness(1, 0), constant(R_MINUS_TWO)), // 2
            mul(fixed(0), exp(2)),                     // 3
            neg(witness(0, 0)),                        // 4
            mul(exp(4), witness(1, 0)),                // 5
            add(exp(5), witness(1, 1)),                // 6
            mul(witness(0, 1), witness(0, 0)),         // 7
            neg(exp(7)),                               // 8
            add(exp(8), witness(1, 0)),                // 9
        ],
        vec![every_row(1), every_row(3), every_row(6), every_row(9)],
    )
}

/// `x0·x1·x2·x3 + x4·x5·x6·x7`, every product a named expression: degree 4. At degree 4 it needs
/// no im pols (`qDeg = 3`); at degree 3 the two cubes become im pols (`2 + 2`).
fn sum_of_quartics_pilout() -> pb::PilOut {
    pilout(
        &["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"],
        &[],
        vec![
            mul(witness(0, 0), witness(1, 0)), // 0: degree 2
            mul(exp(0), witness(2, 0)),        // 1: degree 3
            mul(exp(1), witness(3, 0)),        // 2: degree 4
            mul(witness(4, 0), witness(5, 0)), // 3: degree 2
            mul(exp(3), witness(6, 0)),        // 4: degree 3
            mul(exp(4), witness(7, 0)),        // 5: degree 4
            add(exp(2), exp(5)),               // 6: degree 4
        ],
        vec![every_row(6)],
    )
}

// ---------------------------------------------------------------------------
// What the passes produce
// ---------------------------------------------------------------------------

fn run(pilout: &pb::PilOut, cfg: &PilInfoCfg) -> PilInfoResult {
    pil_info::run(pilout, 0, 0, cfg, &Default::default()).unwrap()
}

/// Every code block the passes emit: expressions, constraints and the verifier's.
fn all_code(result: &PilInfoResult) -> Vec<&CodeEntry> {
    let info = &result.pil_code.expressions_info;
    let verifier = &result.pil_code.verifier_info;
    info.expressions_code
        .iter()
        .flat_map(|e| &e.code)
        .chain(info.constraints.iter().flat_map(|c| &c.code))
        .chain(&verifier.q_verifier.code)
        .chain(verifier.query_verifier.iter().flat_map(|q| &q.code))
        .collect()
}

/// The values of the `number` operands in the emitted code.
fn numbers(result: &PilInfoResult) -> Vec<String> {
    all_code(result)
        .into_iter()
        .flat_map(|c| std::iter::once(&c.dest).chain(&c.src))
        .filter(|r| r.ref_type == "number")
        .filter_map(|r| r.value.clone())
        .collect()
}

fn number_values_in_arena(result: &PilInfoResult) -> Vec<String> {
    fn walk(e: &pil_info::expr::expression::Expression, out: &mut Vec<String>) {
        if e.op == "number" {
            out.extend(e.value.clone());
        }
        for child in &e.values {
            if let pil_info::expr::expression::ExprChild::Inline(inline) = child {
                walk(inline, out);
            }
        }
    }
    let mut out = Vec::new();
    for e in &result.setup.expressions {
        walk(e, &mut out);
    }
    out
}

fn im_pols(result: &PilInfoResult) -> usize {
    result.setup.cm_pols_map.iter().filter(|p| p.im_pol).count()
}

/// Everything criterion 2 of M10 asks of a pilfflonk run: dimension 1 everywhere, and no trace of
/// the FRI opening.
fn assert_bn254_shape(result: &PilInfoResult) {
    let setup = &result.setup;
    let q_stage = setup.n_stages + 1;

    for s in &setup.symbols {
        assert_eq!(s.dim, 1, "symbol {} ({})", s.name, s.sym_type);
    }
    for (name, map) in [("cmPolsMap", &setup.cm_pols_map), ("challengesMap", &setup.challenges_map)] {
        for p in map {
            assert_eq!(p.dim, 1, "{name}: {}", p.name);
        }
    }
    for e in &setup.expressions {
        assert!(e.dim <= 1, "expression {} of dim {}", e.op, e.dim);
    }
    for c in all_code(result) {
        for r in std::iter::once(&c.dest).chain(&c.src) {
            assert_eq!(r.dim, 1, "{} {} in {} code", r.ref_type, r.id, c.op);
        }
    }

    // No FRI: no FRI polynomial, no queryVerifier, no FRI challenges, no quotient evaluations.
    assert_eq!(result.fri_exp_id, None);
    assert_eq!(result.pil_code.fri_exp_id, None);
    assert!(result.pil_code.verifier_info.query_verifier.is_none());
    assert!(result.pil_code.challenges_map.is_empty());
    let challenges: Vec<&str> = setup.challenges_map.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(challenges, ["std_vc", "std_xi"]);
    for e in &setup.expressions {
        assert!(!matches!(e.op.as_str(), "xDivXSubXi" | "eval"), "a FRI node: {}", e.op);
    }
    for c in all_code(result) {
        assert_ne!(c.dest.ref_type, "f", "the FRI polynomial's destination");
        assert!(c.src.iter().all(|r| r.ref_type != "xDivXSubXi"));
    }
    for ev in &result.pil_code.ev_map {
        let stage = setup.cm_pols_map.get(ev.id).and_then(|p| p.stage);
        assert!(ev.entry_type != "cm" || stage != Some(q_stage), "quotient piece {} in the evMap", ev.id);
    }
    let verifier_info = serde_json::to_value(&result.pil_code.verifier_info).expect("serializable");
    let keys: Vec<&String> = verifier_info.as_object().expect("an object").keys().collect();
    assert_eq!(keys, ["qVerifier"]);
    assert_eq!(build_verifier_info_json(&result.pil_code.verifier_info), verifier_info);
}

// ---------------------------------------------------------------------------
// Tests on pilouts built in code
// ---------------------------------------------------------------------------

#[test]
fn bn254_has_dimension_one_and_no_fri() {
    let result = run(&wide_constants_pilout(), &PilInfoCfg::bn254());
    assert_bn254_shape(&result);
}

#[test]
fn bn254_keeps_wide_constants_and_negates_with_r_minus_one() {
    let result = run(&wide_constants_pilout(), &PilInfoCfg::bn254());

    let in_code = numbers(&result);
    for value in [WIDE, R_MINUS_TWO, R_MINUS_ONE] {
        assert!(in_code.iter().any(|n| n == value), "{value} not in the code: {in_code:?}");
    }
    assert!(!in_code.iter().any(|n| n == GOLDILOCKS_NEG_ONE));

    // Both negations became multiplications by r − 1.
    let in_arena = number_values_in_arena(&result);
    assert_eq!(in_arena.iter().filter(|n| *n == R_MINUS_ONE).count(), 2, "{in_arena:?}");
}

/// The same pilout under the STARK's configuration still gets the FRI opening.
#[test]
fn goldilocks_keeps_the_fri_opening() {
    let pilout = wide_constants_pilout();
    let result = run(&pilout, &PilInfoCfg::goldilocks(1));

    let fri_exp_id = result.fri_exp_id.expect("friExpId");
    assert_eq!(result.pil_code.fri_exp_id, Some(fri_exp_id));
    let query_verifier = result.pil_code.verifier_info.query_verifier.as_ref().expect("queryVerifier");
    assert_eq!(query_verifier.exp_id, fri_exp_id);
    let names: Vec<&str> = result.pil_code.challenges_map.iter().map(|c| c.name.as_str()).collect();
    assert!(names.contains(&"std_vf1") && names.contains(&"std_vf2"), "{names:?}");
    let verifier_info = serde_json::to_value(&result.pil_code.verifier_info).expect("serializable");
    assert!(verifier_info.get("queryVerifier").is_some());

    let q_stage = result.setup.n_stages + 1;
    let q_evals = result
        .pil_code
        .ev_map
        .iter()
        .filter(|ev| ev.entry_type == "cm" && result.setup.cm_pols_map[ev.id].stage == Some(q_stage))
        .count();
    assert_eq!(q_evals as i64, result.q_deg, "one evaluation per quotient piece");

    assert!(numbers(&result).iter().any(|n| n == GOLDILOCKS_NEG_ONE));
    assert!(result.setup.cm_pols_map.iter().any(|p| p.dim == 3), "the quotient lives in the extension");
}

#[test]
fn bn254_default_search_needs_no_im_pols_for_a_quartic() {
    let result = run(&sum_of_quartics_pilout(), &PilInfoCfg::bn254());
    assert_bn254_shape(&result);
    assert_eq!(im_pols(&result), 0);
    assert_eq!(result.q_deg, 3);
}

#[test]
fn bn254_lower_max_constraint_degree_adds_im_pols() {
    let cfg = PilInfoCfg { degree_policy: DegreePolicy::Search { max: 3 }, ..PilInfoCfg::bn254() };
    let result = run(&sum_of_quartics_pilout(), &cfg);
    assert_bn254_shape(&result);
    assert_eq!(im_pols(&result), 2);
    assert_eq!(result.q_deg, 2);
    // The im pols live at the last stage of the air (spec §4.2.3): stage 1 here.
    for p in result.setup.cm_pols_map.iter().filter(|p| p.im_pol) {
        assert_eq!(p.stage, Some(result.setup.n_stages));
    }
}

// ---------------------------------------------------------------------------
// Tests on compiled PIL (need PIL2C_EXEC)
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repository root")
}

/// Compile `pil` (relative to the repository root) over BN254 with `PIL2C_EXEC`.
fn compile_bn254(pil: &str) -> pb::PilOut {
    let compiler = std::env::var("PIL2C_EXEC")
        .expect("PIL2C_EXEC must name a pil2com that honours `prime` (e.g. <pil2-compiler>/src/pil.js)");
    let root = repo_root();
    let stem = Path::new(pil).file_stem().expect("a file name").to_string_lossy().into_owned();
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{stem}.bn254.pilout"));
    let status = Command::new(compiler)
        .current_dir(&root)
        .arg(pil)
        .arg("-I")
        .arg("pil2-components/lib/std/pil")
        .arg("-P")
        .arg("pilfflonk/tests/fixtures/fibonacci/bn254.json")
        .arg("-o")
        .arg(&out)
        .status()
        .expect("PIL2C_EXEC runs");
    assert!(status.success(), "pil2com failed on {pil}");
    let pilout = PilOutProxy::new(out.to_str().expect("a UTF-8 path")).expect("a pilout").pilout;
    assert_eq!(pilout.base_field, big_be(R), "{pil} was not compiled over BN254: does PIL2C_EXEC honour `prime`?");
    pilout
}

/// The M13 Fibonacci fixture (5 `everyRow` constraints of degree up to 3, offsets {0, 1}).
///
/// Its degrees tie: at degree 2, `l1' − next` becomes an im pol and `qDeg = 1` (`1 + 1`); at
/// degree 3 there are no im pols and `qDeg = 2` (`0 + 2`). The search keeps the lowest degree on a
/// tie, as pil-stark's does (D5), so the result is one im pol and `qDeg = 1`, not the `qDeg = 2`
/// without im pols that plan §2.1 expects.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn fibonacci_fixture_over_bn254() {
    let pilout = compile_bn254("pilfflonk/tests/fixtures/fibonacci/fibonacci.pil");
    let result = run(&pilout, &PilInfoCfg::bn254());

    assert_bn254_shape(&result);
    let user_constraints = result.setup.constraints.iter().filter(|c| !c.im_pol).count();
    assert_eq!(user_constraints, 5);
    assert!(result.setup.constraints.iter().all(|c| c.boundary == "everyRow"));
    assert_eq!(result.setup.opening_points, [0, 1]);
    assert_eq!(im_pols(&result) as i64 + result.q_deg, 2, "the minimum nImPols + qDeg");
    assert_eq!((im_pols(&result), result.q_deg), (1, 1), "the lowest degree wins the tie");
    // The im pol lives at the last stage of the air (spec §4.2.3), stage 1 here.
    for p in result.setup.cm_pols_map.iter().filter(|p| p.im_pol) {
        assert_eq!(p.stage, Some(1));
    }
    let in_code = numbers(&result);
    assert!(!in_code.iter().any(|n| n == GOLDILOCKS_NEG_ONE), "{in_code:?}");

    let at_degree_3 =
        run(&pilout, &PilInfoCfg { degree_policy: DegreePolicy::Search { max: 3 }, ..PilInfoCfg::bn254() });
    assert_eq!((im_pols(&at_degree_3), at_degree_3.q_deg), (1, 1), "the same tie within 2..=3");

    println!(
        "fibonacci: nImPols {} | qDeg {} | cmPolsMap {:?} | evMap {:?} | qVerifier {} ops",
        im_pols(&result),
        result.q_deg,
        result.setup.cm_pols_map.iter().map(|p| (&p.name, p.stage, p.dim)).collect::<Vec<_>>(),
        result.pil_code.ev_map.iter().map(|e| (&e.entry_type, e.id, e.prime)).collect::<Vec<_>>(),
        result.pil_code.verifier_info.q_verifier.code.len(),
    );
}

/// `tests/fixtures/wide_constants.pil` compiled over BN254: the compiler's constants survive whole.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn wide_constants_fixture_over_bn254() {
    let pilout = compile_bn254("setup/pil-info/tests/fixtures/wide_constants.pil");
    let result = run(&pilout, &PilInfoCfg::bn254());

    assert_bn254_shape(&result);
    let in_code = numbers(&result);
    for value in [WIDE, R_MINUS_TWO, R_MINUS_ONE] {
        assert!(in_code.iter().any(|n| n == value), "{value} not in the code: {in_code:?}");
    }
    assert!(!in_code.iter().any(|n| n == GOLDILOCKS_NEG_ONE));
    println!("wide constants: numbers in the code {in_code:?}");
}
